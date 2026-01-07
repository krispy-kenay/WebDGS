export interface AdamHyperparameters {
    lr_pos: number;
    lr_color: number;
    lr_opacity: number;
    lr_scale: number;
    lr_rot: number;
    beta1: number;
    beta2: number;
    epsilon: number;
}

export const DEFAULT_ADAM_HYPERPARAMETERS = {
    lr_pos: 0.00016,
    lr_color: 0.0025,
    lr_opacity: 0.05,
    lr_scale: 0.005,
    lr_rot: 0.001,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
} satisfies AdamHyperparameters;
