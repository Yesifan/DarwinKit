import type { PageLoad } from './$types';
export const ssr = false;
export const load: PageLoad = ({ params }: { params: { models: string } }) => {
	const models = params.models.split('&');
	return { models };
};
