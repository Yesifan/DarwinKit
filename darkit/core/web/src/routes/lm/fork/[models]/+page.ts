import { get } from '$lib/api';
import type { NXNode, NXEdge } from '$lib/components/model-editor.svelte';

const getLoadModelInfo = async (name: string) => {
	const response = await get(`/lm/edit/load/${name}`);
	if (response.status === 200) {
		return response.json() as Promise<{ nodes: NXNode[]; edges: NXEdge[] }>;
	} else {
		return response.json() as Promise<{
			code: number;
			detail: string;
			name: string;
		}>;
	}
};

export function load({ params }) {
	const data = getLoadModelInfo(params.models);
	return {
		name: params.models,
		res: data
	};
}
