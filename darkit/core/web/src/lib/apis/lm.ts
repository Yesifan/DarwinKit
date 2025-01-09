import { get, post } from '$lib/api';

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

/** 获取以及训练完成的模型 */
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

export const getModelsOptions = async () => {
	const res = await get('/lm/models/options');
	return res.json() as Promise<{ [key: string]: { model: any; trainer: any } }>;
};

export const getTrainResources = async () => {
	const res = await get('/lm/train/resources');
	return res.json() as Promise<string[][]>;
};

export const getTrainCommand = async (
	type: string,
	fork: string | null,
	resume: string | null,
	dataset: string,
	tokenizer: string,
	modelOption: any,
	trainerOption: any
) => {
	const res = await post(`/lm/model/train/command`, {
		type: type,
		fork: fork,
		resume: resume,
		dataset: dataset,
		tokenizer: tokenizer,
		m_conf: modelOption,
		t_conf: trainerOption
	});
	return res.text() as Promise<string>;
};

export const startTrainModel = async (
	type: string,
	fork: string | null,
	resume: string | null,
	dataset: string,
	tokenizer: string,
	modelOption: any,
	trainerOption: any
) => {
	const res = await post('/lm/v2/model/train/', {
		type: type,
		fork: fork,
		resume: resume,
		dataset: dataset,
		tokenizer: tokenizer,
		m_conf: modelOption,
		t_conf: trainerOption
	});
	return res.json() as Promise<any>;
};

export const createForkNetwork = async (
	name: string,
	modelType: string,
	modelOption: any,
	trainerOption: any
) => {
	const res = await post(`/lm/edit/init/${name}`, {
		model: modelType,
		m_conf: modelOption,
		t_conf: trainerOption
	});
	return res.json() as Promise<any>;
};
