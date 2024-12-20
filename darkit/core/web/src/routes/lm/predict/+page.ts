import { getTrainedNetworks } from '$lib/apis/lm';

export const load = async () => {
	const networks = await getTrainedNetworks();

	return {
		networks: networks.filter((network) => network.checkpoints.length > 0)
	};
};
