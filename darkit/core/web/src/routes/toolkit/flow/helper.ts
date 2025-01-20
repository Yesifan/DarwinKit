import { type Operator } from '$lib/flow';
import type { Edge, Node } from '@xyflow/svelte';

export type MyNode = Node<{ label: string; type: string; func?: string; params?: string }>;

export const DefaultModelLayers = [
	{
		type: 'input',
		data: { type: 'Input', params: '10, 4, 24, 24' }
	},
	{
		type: 'default',
		data: { type: 'Conv', func: 'layer.conv2d', params: '1, 24, 3, padding=1' }
	},
	{
		type: 'default',
		data: { type: 'Neuron', func: 'neuron.IFNode' }
	},
	{
		type: 'default',
		data: { type: 'Pool', func: 'layer.MaxPool2d', params: '2, 2' }
	},
	{
		type: 'default',
		data: { type: 'Conv', func: 'layer.conv2d', params: '24, 24, 3, padding=1' }
	},
	{
		type: 'default',
		data: { type: 'Neuron', func: 'neuron.IFNode' }
	},
	{
		type: 'default',
		data: { type: 'Pool', func: 'layer.MaxPool2d', params: '2, 2' }
	},
	{
		type: 'default',
		data: { type: 'Flatten', func: 'layer.Flatten' }
	},
	{
		type: 'default',
		data: { type: 'Linear', func: 'layer.Linear', params: '1176, 384' }
	},
	{
		type: 'default',
		data: { type: 'Neuron', func: 'neuron.IFNode' }
	},
	{
		type: 'default',
		data: { type: 'Linear', func: 'layer.Linear', params: '384, 10' }
	},
	{
		type: 'default',
		data: { type: 'Neuron', func: 'neuron.IFNode' }
	}
];

export const getDefaultModelsNodesEdges = () => {
	const nodes: MyNode[] = DefaultModelLayers.map((layer, index) => {
		const newNode = type2node(layer.data.type as Operator, index, [0, index * 80]);
		newNode.data.label = `${layer.data.func ?? layer.data.type}(${layer.data.params ?? ''})`;
		return newNode;
	});
	const edges: Edge[] = nodes.reduce((acc, target, index) => {
		if (index > 0) {
			const source = nodes[index - 1];
			const edge = {
				id: `${source.id}-${target.id}`,
				source: source.id,
				target: target.id
			};
			acc.push(edge);
		}
		return acc;
	}, [] as Edge[]);
	return [nodes, edges] as [MyNode[], Edge[]];
};

export const type2node = (type: Operator, index: number, [x, y] = [0, 0]) => {
	// 输入的名称是固定的
	const target = type === 'Input' ? 'input' : `${type}-${index++}`;
	return {
		id: target,
		type: 'default',
		data: { label: target, type },
		position: { x: x, y: y + 80 },
		deletable: true
	};
};
