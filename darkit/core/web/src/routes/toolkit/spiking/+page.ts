import { superValidate } from 'sveltekit-superforms';
import { zod } from 'sveltekit-superforms/adapters';
import { uploadFormSchema, controlFormSchema, trainFormSchema } from './schema';

export const load = async () => {
	return {
		uploadForm: await superValidate(zod(uploadFormSchema)),
		controlForm: await superValidate(zod(controlFormSchema)),
		trainForm: await superValidate(zod(trainFormSchema))
	};
};
