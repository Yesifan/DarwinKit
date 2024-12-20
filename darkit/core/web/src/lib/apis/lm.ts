import { get } from '$lib/api';

export type ModelLabel = {
	default: string | number;
	type: string;
	comment: string;
	required: boolean;
};
export interface ModelOption {
	model: Record<string, ModelLabel>;
	trainer: Record<string, ModelLabel>;
}
export const getModelOptions = async () => {
	const res = await get('/lm/models/options');
	return res.json() as Promise<Record<string, ModelOption>>;
};

type ExternalConfig = { fork?: string; series: any };
export interface TrainedModel {
	name: string;
	checkpoints: string[];
	config: ExternalConfig;
}

export const getTrainedNetworks = async () => {
	const res = await get('/lm/models');
	return res.json() as Promise<TrainedModel[]>;
};

export interface ModelDetail {
	model_name: string;
	model_config: Record<string, any>;
	external_config: ExternalConfig;
}
export const getModelDetail = async (model: string) => {
	const res = await get(`/lm/models/${model}`);
	return res.json() as Promise<TrainedModel>;
};
