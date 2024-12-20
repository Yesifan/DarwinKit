import { get } from '$lib/api';
import { getModelOptions } from '$lib/apis/lm';

export interface ForkModel {
	name: string;
	model: string;
	m_conf: any;
	t_conf: any;
}

export async function load() {
	const models: ForkModel[] = await get('/lm/fork/models').then((res) => res.json());
	const resources: [string[], string[]] = await get('/lm/train/resources').then((res) =>
		res.json()
	);
	const modelOptions = await getModelOptions();
	const [datasetList, tokenizerList] = resources;

	return { forkModels: models, datasetList, tokenizerList, options: modelOptions };
}
