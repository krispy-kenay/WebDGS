
export interface LoadedImage {
    name: string;
    file: File;
    bitmap: ImageBitmap;
    width: number;
    height: number;
    texture: GPUTexture;
}

export async function loadImages(files: FileList | File[], device: GPUDevice): Promise<LoadedImage[]> {
    const fileArray = Array.from(files).filter(f => {
        const name = f.name.toLowerCase();
        return name.endsWith('.jpg') || name.endsWith('.jpeg') || name.endsWith('.png');
    });

    fileArray.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }));

    const promises = fileArray.map(async (file) => {
        try {
            const bitmap = await createImageBitmap(file);
            const texture = createTextureFromImage(device, bitmap);
            return {
                name: file.name,
                file: file,
                bitmap: bitmap,
                width: bitmap.width,
                height: bitmap.height,
                texture: texture,
            } as LoadedImage;
        } catch (e) {
            console.error(`Failed to load image ${file.name}:`, e);
            return null;
        }
    });

    const results = await Promise.all(promises);

    return results.filter((img): img is LoadedImage => img !== null);
}

export function createTextureFromImage(device: GPUDevice, image: ImageBitmap): GPUTexture {
    const texture = device.createTexture({
        size: [image.width, image.height, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    device.queue.copyExternalImageToTexture(
        { source: image },
        { texture: texture },
        [image.width, image.height]
    );

    return texture;
}
