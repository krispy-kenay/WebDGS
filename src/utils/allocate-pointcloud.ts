import { PointCloud } from './load';

export interface AllocatePointCloudLikeOptions {
  numPoints: number;
  usage?: GPUBufferUsageFlags;
}

export function allocatePointCloudLike(
  device: GPUDevice,
  template: PointCloud,
  options: AllocatePointCloudLikeOptions
): PointCloud {
  const numPoints = Math.max(1, Math.floor(options.numPoints));
  const templatePoints = Math.max(1, Math.floor(template.num_points));

  const bytesPerGaussian = Math.max(1, Math.floor(template.gaussian_3d_buffer.size / templatePoints));
  const usage =
    options.usage ??
    (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);

  const gaussian_3d_buffer = device.createBuffer({
    label: 'resized gaussian_3d_buffer',
    size: numPoints * bytesPerGaussian,
    usage,
  });

  let sh_buffer: GPUBuffer | undefined;
  if (template.sh_buffer) {
    const bytesPerSh = Math.max(1, Math.floor(template.sh_buffer.size / templatePoints));
    sh_buffer = device.createBuffer({
      label: 'resized sh_buffer',
      size: numPoints * bytesPerSh,
      usage,
    });
  }

  return {
    type: (template as any).type ?? 'normal',
    num_points: numPoints,
    sh_deg: template.sh_deg ?? 0,
    gaussian_3d_buffer,
    ...(sh_buffer ? { sh_buffer } : {}),
  };
}

