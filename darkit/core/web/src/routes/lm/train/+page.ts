import { getModelsOptions, getTrainResources, getTrainedNetworks } from '$lib/apis/lm';

export const load = async () => {
	const resources = await getTrainResources();
	const modelsOptions = await getModelsOptions();
	const trainedNetworks = await getTrainedNetworks();

	return {
		resources,
		modelsOptions,
		trainedNetworks
	};
};
