import { Float16Array } from '@petamoriken/float16';
import { log, time, timeLog } from './simple-console';
import { decodeHeader, readRawVertex, nShCoeffs } from './plyreader';

const c_size_float = 2;

const c_size_3d_gaussian =
    3 * c_size_float
    + c_size_float
    + 4 * c_size_float
    + 4 * c_size_float
    ;

export type PointCloudType = 'full' | 'normal';

export interface PointCloud {
    type: PointCloudType;
    num_points: number;
    sh_deg?: number;
    gaussian_3d_buffer: GPUBuffer;
    sh_buffer?: GPUBuffer;
    color_buffer?: GPUBuffer;
}

const yielder = () => new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
});

export async function loadPointCloud(file: BlobPart | File, device: GPUDevice): Promise<PointCloud> {
    const blob = file instanceof Blob ? file : new Blob([file]);
    const arrayBuffer = await new Promise<ArrayBuffer>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => resolve(event.target?.result as ArrayBuffer);
        reader.onerror = reject;
        reader.readAsArrayBuffer(blob);
    });

    // Check if it's a PLY file
    const headerCheck = new Uint8Array(arrayBuffer.slice(0, 4));
    const isPly = headerCheck[0] === 0x70 && headerCheck[1] === 0x6c && headerCheck[2] === 0x79; // 'ply'

    if (isPly) {
        return loadPly(arrayBuffer, device);
    }

    // Try parsing as COLMAP .bin
    try {
        return await loadColmapBin(arrayBuffer, device);
    } catch (e) {
        throw new Error(`Failed to load pointcloud: ${(e as Error).message}`);
    }
}

async function loadColmapBin(arrayBuffer: ArrayBuffer, device: GPUDevice): Promise<PointCloud> {
    const view = new DataView(arrayBuffer);
    let offset = 0;

    const num_points_big = view.getBigUint64(offset, true);
    const num_points = Number(num_points_big);
    offset += 8;

    log(`num points: ${num_points} (from .bin)`);
    log(`processing loaded attributes...`);
    time();

    const gaussian_3d_buffer = device.createBuffer({
        label: 'bin input 3d gaussians data buffer',
        size: num_points * 24,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    const gaussian = new Float16Array(gaussian_3d_buffer.getMappedRange());

    const sh_buffer = device.createBuffer({
        label: 'bin sh/color data buffer',
        size: num_points * 3 * 16 * 2,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    const sh = new Float16Array(sh_buffer.getMappedRange());

    const YIELD_STRIDE = 2000;
    const C0 = 0.28209479177387814;

    for (let i = 0; i < num_points; i++) {
        offset += 8;

        const x = view.getFloat64(offset, true); offset += 8;
        const y = view.getFloat64(offset, true); offset += 8;
        const z = view.getFloat64(offset, true); offset += 8;

        const r_u8 = view.getUint8(offset); offset += 1;
        const g_u8 = view.getUint8(offset); offset += 1;
        const b_u8 = view.getUint8(offset); offset += 1;

        // error
        offset += 8;

        const track_len_big = view.getBigUint64(offset, true);
        const track_len = Number(track_len_big);
        offset += 8;

        // Skip track
        offset += track_len * 8;

        // Fill buffers
        const o = i * 12;

        gaussian[o + 0] = x;
        gaussian[o + 1] = y;
        gaussian[o + 2] = z;
        // Defaults for normal point cloud
        gaussian[o + 3] = 1.0;
        gaussian[o + 4] = 1.0;
        gaussian[o + 5] = 0.0;
        gaussian[o + 6] = 0.0;
        gaussian[o + 7] = 0.0;
        gaussian[o + 8] = -5.0;
        gaussian[o + 9] = -5.0;
        gaussian[o + 10] = -5.0;
        gaussian[o + 11] = 0.0;

        // Color to SH DC
        const r = r_u8 / 255.0;
        const g = g_u8 / 255.0;
        const b = b_u8 / 255.0;

        const dc_r = (r - 0.5) / C0;
        const dc_g = (g - 0.5) / C0;
        const dc_b = (b - 0.5) / C0;

        const output_offset = i * 16 * 3;
        sh[output_offset + 0] = dc_r;
        sh[output_offset + 1] = dc_g;
        sh[output_offset + 2] = dc_b;

        if (i % YIELD_STRIDE === 0) {
            // Need to yield occasionally to keep UI responsive
            await yielder();
        }
    }

    gaussian_3d_buffer.unmap();
    sh_buffer.unmap();
    timeLog();

    return {
        type: 'normal',
        num_points,
        sh_deg: 0,
        gaussian_3d_buffer,
        sh_buffer
    };
}

async function loadPly(arrayBuffer: ArrayBuffer, device: GPUDevice): Promise<PointCloud> {
    const [vertexCount, propertyTypes, vertexData] = decodeHeader(arrayBuffer);

    // Detect if full or normal
    const hasRot = 'rot_0' in propertyTypes;
    const hasScale = 'scale_0' in propertyTypes;
    const isFull = hasRot && hasScale;
    const type: PointCloudType = isFull ? 'full' : 'normal';

    // For Full: SH logic
    let sh_deg = 0;
    let num_coefs = 0;
    let max_num_coefs = 0;
    let nCoeffsPerColor = 0;
    let shFeatureOrder: string[] = [];

    if (isFull) {
        let nRestCoeffs = 0;
        for (const propertyName in propertyTypes) {
            if (propertyName.startsWith('f_rest_')) {
                nRestCoeffs += 1;
            }
        }
        nCoeffsPerColor = nRestCoeffs / 3;
        sh_deg = Math.sqrt(nCoeffsPerColor + 1) - 1;
        num_coefs = nShCoeffs(sh_deg);
        max_num_coefs = 16;

        for (let rgb = 0; rgb < 3; ++rgb) {
            shFeatureOrder.push(`f_dc_${rgb}`);
        }
        for (let i = 0; i < nCoeffsPerColor; ++i) {
            for (let rgb = 0; rgb < 3; ++rgb) {
                shFeatureOrder.push(`f_rest_${rgb * nCoeffsPerColor + i}`);
            }
        }
    }

    const c_size_sh_coef = 3 * max_num_coefs * c_size_float;

    const num_points = vertexCount;

    log(`num points: ${num_points}, type: ${type}`);
    log(`processing loaded attributes...`);
    time();

    const gaussian_3d_buffer = device.createBuffer({
        label: 'ply input 3d gaussians data buffer',
        size: num_points * c_size_3d_gaussian,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    const gaussian = new Float16Array(gaussian_3d_buffer.getMappedRange());

    let sh_buffer: GPUBuffer;
    let sh: Float16Array;


    const sh_buffer_size = isFull ? num_points * c_size_sh_coef : num_points * 3 * 16 * c_size_float;
    sh_buffer = device.createBuffer({
        label: 'ply sh/color data buffer',
        size: num_points * 3 * 16 * c_size_float,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    sh = new Float16Array(sh_buffer.getMappedRange());

    let readOffset = 0;
    const YIELD_STRIDE = 2000;

    for (let i = 0; i < num_points; i++) {
        const [newReadOffset, rawVertex] = readRawVertex(readOffset, vertexData, propertyTypes);
        readOffset = newReadOffset;

        const o = i * (c_size_3d_gaussian / c_size_float);

        // Fill Gaussian 3D
        gaussian[o + 0] = rawVertex.x;
        gaussian[o + 1] = rawVertex.y;
        gaussian[o + 2] = rawVertex.z;

        if (isFull) {
            gaussian[o + 3] = rawVertex.opacity;
            gaussian[o + 4] = rawVertex.rot_0;
            gaussian[o + 5] = rawVertex.rot_1;
            gaussian[o + 6] = rawVertex.rot_2;
            gaussian[o + 7] = rawVertex.rot_3;
            gaussian[o + 8] = rawVertex.scale_0;
            gaussian[o + 9] = rawVertex.scale_1;
            gaussian[o + 10] = rawVertex.scale_2;

            const output_offset = i * 16 * 3;
            for (let order = 0; order < num_coefs; ++order) {
                const order_offset = order * 3;
                for (let j = 0; j < 3; ++j) {
                    const coeffName = shFeatureOrder[order * 3 + j];
                    sh[output_offset + order_offset + j] = rawVertex[coeffName];
                }
            }
        } else {
            // Normal Point Cloud
            gaussian[o + 3] = 1.0;
            gaussian[o + 4] = 1.0;
            gaussian[o + 5] = 0.0;
            gaussian[o + 6] = 0.0;
            gaussian[o + 7] = 0.0;
            gaussian[o + 8] = -5.0;
            gaussian[o + 9] = -5.0;
            gaussian[o + 10] = -5.0;

            const C0 = 0.28209479177387814;
            let r, g, b;
            if ('red' in rawVertex) {
                r = rawVertex.red / 255.0;
                g = rawVertex.green / 255.0;
                b = rawVertex.blue / 255.0;
            } else if ('diffuse_red' in rawVertex) {
                r = rawVertex.diffuse_red / 255.0;
                g = rawVertex.diffuse_green / 255.0;
                b = rawVertex.diffuse_blue / 255.0;
            } else {
                r = 0.5; g = 0.5; b = 0.5;
            }

            const dc_r = (r - 0.5) / C0;
            const dc_g = (g - 0.5) / C0;
            const dc_b = (b - 0.5) / C0;

            const output_offset = i * 16 * 3;
            sh[output_offset + 0] = dc_r;
            sh[output_offset + 1] = dc_g;
            sh[output_offset + 2] = dc_b;
        }

        if (i % YIELD_STRIDE === 0) {
            await yielder();
        }
    }

    gaussian_3d_buffer.unmap();
    sh_buffer.unmap();

    timeLog();

    return {
        type,
        num_points,
        sh_deg: isFull ? sh_deg : 0,
        gaussian_3d_buffer,
        sh_buffer,
    };
}
