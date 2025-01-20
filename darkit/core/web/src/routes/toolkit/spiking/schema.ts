import { z } from 'zod';

export const Modules = [
	{ label: 'IF', value: 'IF' },
	{ label: 'LIF', value: 'LIF' }
];

export const uploadFormSchema = z.object({
	model_name: z.string().min(2).max(50).default('ANNModel'),
	file: z
		.instanceof(File, { message: 'Please upload a file.' })
		.refine((f) => f.size < 100_000, 'Max 100 kB upload size.')
});

export const controlFormSchema = z.object({
	type: z.string().default('LIF'),
	v_threshold: z.number().min(0).default(0.9)
});

export const trainFormSchema = z.object({
	name: z.string().min(2).max(50).default('MyModel'),
	device: z.enum(['cpu', 'cuda']).default('cpu'),
	max_step: z.number().int().min(1).default(1000),
	batch_size: z.number().int().min(1).default(4),
	T: z.number().int().min(0).default(10),
	learning_rate: z.number().min(0).default(0.01),
	save_step_interval: z.number().int().min(1).default(1000)
});

export type UploadFormSchema = typeof uploadFormSchema;
export type ControlFormSchema = typeof controlFormSchema;
export type TrainFormSchema = typeof trainFormSchema;
