import { Mat4, mat4, Vec3, vec3, Quat, quat } from 'wgpu-matrix';
import { log, time, timeLog } from './simple-console';

export interface CameraData {
    id: number;
    // Extrinsics
    position?: Vec3;
    rotation?: Mat4;
    // Intrinsics
    width?: number;
    height?: number;
    fx?: number;
    fy?: number;
    cx?: number;
    cy?: number;
    // Metadata
    img_name?: string;
    camera_id?: number;
}

const yielder = () => new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
});

export async function loadCamera(fileOrFiles: BlobPart | File | File[]): Promise<CameraData[]> {
    let files: File[] = [];
    if (Array.isArray(fileOrFiles)) {
        files = fileOrFiles;
    } else if (fileOrFiles instanceof File) {
        files = [fileOrFiles];
    } else {
        return loadSingleCameraFile(fileOrFiles);
    }

    // Identify files
    const imagesVal = files.find(f => f.name.toLowerCase().endsWith('images.bin'));
    const camerasVal = files.find(f => f.name.toLowerCase().endsWith('cameras.bin'));
    const jsonVal = files.find(f => f.name.toLowerCase().endsWith('.json'));

    if (jsonVal) {
        return loadSingleCameraFile(jsonVal);
    }

    if (imagesVal && camerasVal) {
        log('Loading merged COLMAP data (images.bin + cameras.bin)...');
        const imagesBuffer = await imagesVal.arrayBuffer();
        const camerasBuffer = await camerasVal.arrayBuffer();

        const imagesData = await loadColmapImagesBin(imagesBuffer);
        const camerasData = await loadColmapCamerasBin(camerasBuffer);

        // imagesData has Extrinsics and camera_id.
        // camerasData has Intrinsics and camera_id (stored as 'id').

        // Map for quick lookup of intrinsics
        const cameraMap = new Map<number, CameraData>();
        for (const cam of camerasData) {
            cameraMap.set(cam.id, cam);
        }

        // Merge
        const merged: CameraData[] = [];
        for (const img of imagesData) {
            if (img.camera_id !== undefined && cameraMap.has(img.camera_id)) {
                const intrinsics = cameraMap.get(img.camera_id)!;
                merged.push({
                    ...img,
                    ...intrinsics,
                    id: img.id,
                });
            } else {
                merged.push(img);
            }
        }
        log(`Merged ${merged.length} cameras.`);
        return merged;

    } else if (imagesVal) {
        log('WARNING: Loading images.bin ONLY. Intrinsics (width, height, fx, fy) will be missing.');
        return loadSingleCameraFile(imagesVal);
    } else if (camerasVal) {
        log('WARNING: Loading cameras.bin ONLY. Extrinsics (position, rotation) will be missing.');
        return loadSingleCameraFile(camerasVal);
    } else if (files.length > 0) {
        return loadSingleCameraFile(files[0]);
    }

    return [];
}

async function loadSingleCameraFile(file: BlobPart | File): Promise<CameraData[]> {
    const blob = file instanceof Blob ? file : new Blob([file]);

    // Check filename if available
    let filename = '';
    if (file instanceof File) {
        filename = file.name.toLowerCase();
    }

    const arrayBuffer = await blob.arrayBuffer();

    if (filename.endsWith('.json') || isJson(arrayBuffer)) {
        return loadCameraJson(arrayBuffer);
    } else if (filename.endsWith('images.bin')) {
        return loadColmapImagesBin(arrayBuffer);
    } else if (filename.endsWith('cameras.bin')) {
        return loadColmapCamerasBin(arrayBuffer);
    }

    throw new Error(`Unsupported camera file format: ${filename}`);
}

function isJson(buffer: ArrayBuffer): boolean {
    // Check first few bytes for JSON start characters
    const view = new Uint8Array(buffer.slice(0, 10));
    for (let i = 0; i < view.length; i++) {
        const char = String.fromCharCode(view[i]);
        if (char === '{' || char === '[') return true;
        if (char.trim() !== '') return false;
    }
    return false;
}

// JSON loader
interface CameraJson {
    id: number
    img_name: string
    width: number
    height: number
    position: number[]
    rotation: number[][]
    fx: number
    fy: number
};

async function loadCameraJson(buffer: ArrayBuffer): Promise<CameraData[]> {
    const text = new TextDecoder().decode(buffer);
    const json = JSON.parse(text);

    const list = Array.isArray(json) ? json : [json];
    log(`loaded cameras count (json): ${list.length}`);

    return list.map((j: CameraJson) => {
        const flatRot = j.rotation.flat();

        const R = mat4.identity();

        const r = j.rotation;
        const rotMat = mat4.identity();

        rotMat[0] = r[0][0]; rotMat[1] = r[1][0]; rotMat[2] = r[2][0]; rotMat[3] = 0;
        rotMat[4] = r[0][1]; rotMat[5] = r[1][1]; rotMat[6] = r[2][1]; rotMat[7] = 0;
        rotMat[8] = r[0][2]; rotMat[9] = r[1][2]; rotMat[10] = r[2][2]; rotMat[11] = 0;
        rotMat[12] = 0; rotMat[13] = 0; rotMat[14] = 0; rotMat[15] = 1;

        return {
            id: j.id,
            img_name: j.img_name,
            width: j.width,
            height: j.height,
            fx: j.fx,
            fy: j.fy,
            position: vec3.create(j.position[0], j.position[1], j.position[2]),
            rotation: rotMat
        };
    });
}

// COLMAP images.bin loader
async function loadColmapImagesBin(buffer: ArrayBuffer): Promise<CameraData[]> {
    const view = new DataView(buffer);
    let offset = 0;

    // Safety check size
    if (buffer.byteLength < 8) return [];

    const num_reg_images = Number(view.getBigUint64(offset, true));
    offset += 8;

    const cameras: CameraData[] = [];

    for (let i = 0; i < num_reg_images; i++) {
        const image_id = view.getUint32(offset, true); offset += 4;

        const qw = view.getFloat64(offset, true); offset += 8;
        const qx = view.getFloat64(offset, true); offset += 8;
        const qy = view.getFloat64(offset, true); offset += 8;
        const qz = view.getFloat64(offset, true); offset += 8;

        const tx = view.getFloat64(offset, true); offset += 8;
        const ty = view.getFloat64(offset, true); offset += 8;
        const tz = view.getFloat64(offset, true); offset += 8;

        const camera_id = view.getUint32(offset, true); offset += 4;

        // Name
        let name = "";
        while (true) {
            const charCode = view.getUint8(offset);
            offset += 1;
            if (charCode === 0) break;
            name += String.fromCharCode(charCode);
        }

        // Points2D
        const num_points2d = Number(view.getBigUint64(offset, true));
        offset += 8;

        offset += num_points2d * (24);

        const q = quat.create(qx, qy, qz, qw);
        const R_colmap = mat4.fromQuat(q);

        const R_gl = mat4.create();
        mat4.copy(R_colmap, R_gl);

        const Rt = mat4.transpose(R_colmap);
        const T = vec3.create(tx, ty, tz);

        const C = vec3.create();
        const RtT = vec3.transformMat4(T, Rt);
        vec3.scale(RtT, -1, C);

        // Store
        cameras.push({
            id: image_id,
            camera_id: camera_id,
            img_name: name,
            rotation: R_gl,
            position: C
        });

        if (i % 500 === 0) await yielder();
    }

    log(`loaded bin images count: ${cameras.length}`);
    return cameras;
}

// COLMAP cameras.bin loader
async function loadColmapCamerasBin(buffer: ArrayBuffer): Promise<CameraData[]> {
    const view = new DataView(buffer);
    let offset = 0;

    const num_cameras = Number(view.getBigUint64(offset, true));
    offset += 8;

    const cameras: CameraData[] = [];

    for (let i = 0; i < num_cameras; i++) {
        const camera_id = view.getUint32(offset, true); offset += 4;
        const model_id = view.getInt32(offset, true); offset += 4;

        const width = Number(view.getBigUint64(offset, true)); offset += 8;
        const height = Number(view.getBigUint64(offset, true)); offset += 8;

        let fx = 0, fy = 0, cx = 0, cy = 0;

        if (model_id === 0) {
            const f = view.getFloat64(offset, true); offset += 8;
            cx = view.getFloat64(offset, true); offset += 8;
            cy = view.getFloat64(offset, true); offset += 8;
            fx = f;
            fy = f;
        } else if (model_id === 1) {
            fx = view.getFloat64(offset, true); offset += 8;
            fy = view.getFloat64(offset, true); offset += 8;
            cx = view.getFloat64(offset, true); offset += 8;
            cy = view.getFloat64(offset, true); offset += 8;
        } else {
            throw new Error(`Unsupported COLMAP camera model ID: ${model_id}`);
        }

        cameras.push({
            id: camera_id,
            camera_id: camera_id,
            width,
            height,
            fx,
            fy,
            cx,
            cy
        });
    }

    log(`loaded bin cameras count: ${cameras.length}`);
    return cameras;
}
