import { getTrainedNetworks } from '$lib/apis';

export const load = async () => {
	const networks = await getTrainedNetworks();
	return {
		networks
	};
};
