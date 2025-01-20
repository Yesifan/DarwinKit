import { get, post, delete_ } from '$lib/api';

type ExternalConfig = { fork?: string; series: any };
export interface TrainedModel {
	name: string;
	checkpoints: string[];
	config: ExternalConfig;
}

/** 获取以及训练完成的模型 */
export const getTrainedNetworks = async () => {
	const res = await get('/models');
	return res.json() as Promise<TrainedModel[]>;
};

export const getModel = async (name: string) => {
	const res = await get(`/models/${name}`);
	const data: any = await res.json();
	return [res.status, data];
};

export const delModels = async (models: string[]) => {
	const response = await delete_('/models/', { check_models: models });
	return response;
};

export const stopTrain = async (name: string) => {
	const response = await post(`/models/stop/${name}`);
	return response;
};

export const uploadSpikingModel = async (name: string, file: File) => {
	const formdata = new FormData();
	formdata.append('cname', name);
	formdata.append('file', file);
	const res = await post('/spiking/upload', formdata);
	return res.json() as Promise<any>;
};

export const trainSpikingModel = async (name: string, tconf: any, sconf: any) => {
	const res = await post('/spiking/train', { cname: name, tconf, sconf });
	const data: any = await res.json();
	return [res.status, data];
};
