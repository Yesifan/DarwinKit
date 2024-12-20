<script lang="ts" module>
	import type { Node, Edge, XYPosition, NodeTypes } from '@xyflow/svelte';
	import BaseNode, { NODE_COLORS } from './flow/base-node.svelte';
	import ExpandableNode from './flow/expandable-node.svelte';

	export type NXNode = [string, { sub_module_size: number; depth?: number; level?: number }];
	export type NXEdge = [string, string];

	export type Props = {
		class?: string;
		nodes?: NXNode[];
		edges?: NXEdge[];
		onNodeClick?: (event: CustomEvent<{ node: Node; event: MouseEvent | TouchEvent }>) => void;
		onNodeExpand?: (id: string) => void;
	};

	const NODE_TYPES: NodeTypes = {
		base: BaseNode as any,
		expandable: ExpandableNode as any
	};

	const NODE_COLORS_KEY = Object.keys(NODE_COLORS);

	const EDGE_OPT = {
		type: 'default',
		animated: false,
		markerEnd: MarkerType.Arrow
	};

	/**
	 * 获取每个 node 的 depth, depth 取决于 node 在图中最深的深度
	 */
	function getNodeWithDepth(nodes: NXNode[], edges: NXEdge[]): NXNode[] {
		const nodeDepths = new Map<string, number>();

		const dfs = (node: string, level: number) => {
			if (nodeDepths.has(node)) {
				nodeDepths.set(node, Math.max(nodeDepths.get(node)!, level));
			} else {
				nodeDepths.set(node, level);
			}

			edges
				.filter(([_, target]) => target === node)
				.map(([source]) => source)
				.forEach((source) => dfs(source, level + 1));
		};

		nodes.forEach(([id]) => dfs(id, 0));

		return nodes.map(([id, data]) => [id, { ...data, depth: nodeDepths.get(id)! }]);
	}

	// 根据 nodes 的深度进行层次布局
	function hierarchicalLayout(nodes: NXNode[]): Record<string, XYPosition> {
		const depthMap = new Map<number, NXNode[]>();

		nodes.forEach((node) => {
			const depth = node[1].depth || 0;
			if (!depthMap.has(depth)) {
				depthMap.set(depth, []);
			}
			depthMap.get(depth)!.push(node);
		});

		let y = 0;
		const positions: Record<string, XYPosition> = {};

		Array.from(depthMap.entries())
			.sort(([d1], [d2]) => d1 - d2)
			.forEach(([level, nodesAtLevel]) => {
				nodesAtLevel.sort(([idA], [idB]) => idA.localeCompare(idB));
				let x = 0;
				let direction = 1;
				nodesAtLevel.forEach(([id]) => {
					positions[id] = { x: x * direction, y };
					x += direction === 1 ? 200 : 0; // Adjust horizontal spacing as needed
					direction *= -1; // Switch direction
				});
				y -= 100; // Adjust vertical spacing as needed
			});

		return positions;
	}
</script>

<script lang="ts">
	import '@xyflow/svelte/dist/style.css';
	import { mode } from 'mode-watcher';
	import { writable } from 'svelte/store';
	import { SvelteFlow, Controls, Background, BackgroundVariant, MarkerType } from '@xyflow/svelte';

	const flowNodes = writable<Node[]>([]);
	const flowEdges = writable<Edge[]>([]);

	let {
		class: className,
		nodes = [],
		edges = [],
		onNodeClick = () => {},
		onNodeExpand
	}: Props = $props();
	let nodesWithLevels = $derived(getNodeWithDepth(nodes, edges));

	$effect(() => {
		const positionMap = hierarchicalLayout(nodesWithLevels);
		flowNodes.update((flowNodes) => {
			const newNodes: Node[] = nodesWithLevels
				// .filter(([id]) => !flowNodes.find((item) => item.id === id))
				.map(([id, data]) => {
					const color = NODE_COLORS_KEY[(data.level ?? 0) % NODE_COLORS_KEY.length];
					return {
						id: id,
						type: data.sub_module_size > 0 ? 'expandable' : 'base',
						data: { label: id, color, onClick: onNodeExpand, ...data },
						position: positionMap[id],
						deletable: false
					};
				});
			return newNodes;
		});
	});

	$effect(() => {
		flowEdges.update(() => {
			const newEdges = edges.map(([source, target]) => ({
				id: `${source}-${target}`,
				source: source,
				target: target
			}));
			return newEdges;
		});
	});
</script>

<!--
👇 By default, the Svelte Flow container has a height of 100%.
This means that the parent container needs a height to render the flow.
-->
<div class={className}>
	<SvelteFlow
		nodeTypes={NODE_TYPES}
		nodes={flowNodes}
		edges={flowEdges}
		colorMode={$mode}
		snapGrid={[20, 20]}
		defaultEdgeOptions={EDGE_OPT}
		on:nodeclick={onNodeClick}
		fitView
	>
		<Controls position="bottom-right" />
		<Background variant={BackgroundVariant.Dots} />
	</SvelteFlow>
</div>
