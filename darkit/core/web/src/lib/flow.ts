import type { Edge, Node } from '@xyflow/svelte';

export type Operator = 'Input' | 'Conv' | 'Pool' | 'Linear' | 'Neuron';

/**
 * An object representing the details of an input.
 * This can be used to store and manage various properties related to an input.
 */
export const InputDetail = {
	name: 'Input',
	func: [],
	desc: 'The input is used to provide input shape to the network.',
	params: true
};

export const ConvDetail = {
	name: 'Conv',
	func: [
		{ label: 'Conv2D', value: 'layer.conv2d' },
		{ label: 'Conv2D', value: 'layer.conv3d' }
	],
	params: true,
	desc: 'The conv function is used to perform convolution operations on the input data to extract features from the data.  '
};
export const PoolDetail = {
	name: 'Pool',
	func: [
		{ label: 'MaxPool2D', value: 'layer.maxpool2d' },
		{ label: 'AvgPool2D', value: 'layer.avgpool2d' }
	],
	params: true,

	desc: 'The pool function is used to perform pooling operations on the input data to reduce its dimensionality and extract important features.'
};

export const LinearDetail = {
	name: 'Linear',
	func: [{ label: 'Linear', value: 'layer.linear' }],
	desc: 'The linear function is used to perform linear transformations on the input data.',
	params: true
};

export const FlattenDetail = {
	name: 'Flatten',
	func: [{ label: 'Flatten', value: 'layer.flatten' }],
	desc: 'The flatten function is used to flatten the input data into a single dimension.',
	params: false
};

export const NeuronDetail = {
	name: 'Neuron',
	func: [
		{ label: 'LIFNeuron', value: 'neuron.LIFNode' },
		{ label: 'IFNeuron', value: 'neuron.IFNode' }
	],
	desc: 'The neuron function is used to simulate the behavior of biological neurons.',
	params: true
};

export const OperatorDetail = {
	Input: InputDetail,
	Conv: ConvDetail,
	Pool: PoolDetail,
	Linear: LinearDetail,
	Flatten: FlattenDetail,
	Neuron: NeuronDetail
};

export const generateNetwork = (edges: Edge[]) => {
	const network: string[] = [];
	let source = 'input';
	while (true) {
		const current = edges.find((edge) => edge.source === source);
		if (!current) {
			network.push(source);
			break;
		}
		network.push(current.source);
		source = current.target;
	}
	return network;
};

export const generateNetworkCode4Conn = (nodeId: string[], nodes: Node[]) => {
	const connectNodes = nodeId
		.filter((id) => id !== 'input' && id !== 'output')
		.map((id) => {
			return nodes.find((node) => node.id === id);
		});

	const networkCode = `
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

class Model(nn.Module):
  # 这里取决于input那里的维度
  def __init__(self, T=10): 
    super().__init__()
		self.T = T

    self.layers = nn.Sequential(
			${connectNodes
				.map((node) => {
					return `${node?.data.label}`;
				})
				.join(',\n      ')}
    )
				
    functional.set_step_mode(self, 'm')
    
  def forward(self, x):
    # make sure the x  has T dimension
		x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
    x_seq = self.layers(x_seq)
		fr = x_seq.mean(0)
    return fr
  `;
	return networkCode;
};

export const generateNetworkCode4Edge = (edge: Edge[], nodes: Node[]) => {
	const nodeIndex = generateNetwork(edge);
	return generateNetworkCode4Conn(nodeIndex, nodes);
};
