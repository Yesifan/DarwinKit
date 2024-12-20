import { getTrainedNetworks } from '$lib/apis/lm';

export const load = async () => {
	const networks = await getTrainedNetworks();
	return {
		networks
	};
};
