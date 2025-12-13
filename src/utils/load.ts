import { loadPointCloud, PointCloud } from './load-pointcloud';
import { loadCamera, CameraData } from './load-camera';

export type { PointCloud, CameraData };

export async function load(file: BlobPart | File, device: GPUDevice): Promise<PointCloud | CameraData[]> {
  const blob = file instanceof Blob ? file : new Blob([file]);

  // Check file name if available
  let filename = '';
  if (file instanceof File) {
    filename = file.name.toLowerCase();
  }

  // Check header
  const headerBuffer = await blob.slice(0, 50).arrayBuffer();
  const headerView = new Uint8Array(headerBuffer);

  const isPly = headerView[0] === 0x70 && headerView[1] === 0x6c && headerView[2] === 0x79; // 'ply'

  if (isPly || filename.endsWith('.ply')) {
    return loadPointCloud(file, device);
  }

  // Bin check / JSON check
  if (filename.endsWith('points3d.bin')) {
    return loadPointCloud(file, device);
  }
  else if (filename.endsWith('cameras.bin') || filename.endsWith('images.bin') || filename.endsWith('.json')) {
    return loadCamera(file);
  }

  // Fallback try/catch
  try {
    return await loadPointCloud(file, device);
  } catch (e) {
    try {
      return await loadCamera(file);
    } catch (e2) {
      throw new Error(`Failed to load file. Not a valid PointCloud or Camera file.`);
    }
  }
}
