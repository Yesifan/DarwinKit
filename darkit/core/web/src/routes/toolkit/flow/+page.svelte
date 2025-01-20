<script lang="ts" module>
	import type { Snapshot } from '../$types';
	import type { Edge } from '@xyflow/svelte';
	import { type Operator } from '$lib/flow';
</script>

<script lang="ts">
	import '@xyflow/svelte/dist/style.css';
	import { mode } from 'mode-watcher';
	import { writable } from 'svelte/store';
	import { fade, fly } from 'svelte/transition';
	import { SvelteFlow, Controls, Background, BackgroundVariant, MarkerType } from '@xyflow/svelte';
	// 👇 this is important! You need to import the styles for Svelte Flow to work
	import ControlLib from '$lib/components/flow/control-lib.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import Code from '$lib/components/code.svelte';
	import { generateNetwork, generateNetworkCode4Conn } from '$lib/flow';
	import ControlEditor from '$lib/components/flow/control-editor.svelte';
	import { type2node, getDefaultModelsNodesEdges, type MyNode } from './helper';

	const edgeOpt = {
		type: 'default',
		animated: true,
		markerEnd: MarkerType.Arrow
	};

	let index = 100;
	let pyCode = $state('');
	let showCode = $state(false);
	let selected = $state<MyNode | null>(null);

	let connect = $state<string[]>([]);

	const [defaultNodes, defaultEdges] = getDefaultModelsNodesEdges();
	// We are using writables for the nodes and edges to sync them easily. When a user drags a node for example, Svelte Flow updates its position.
	const nodes = writable<MyNode[]>(defaultNodes);

	// same for edges
	const edges = writable<Edge[]>(defaultEdges);

	const apply = (id: string, func?: string, params?: string) => {
		nodes.update((nodes) => {
			const node = nodes.find((n) => n.id === id);
			if (node) {
				const label = `${func ?? node.data.type}(${params})`;
				node.data = {
					...node.data,
					label,
					func,
					params
				};
			}

			return nodes;
		});
	};
	const addNode = (type: Operator) => {
		index += 1;
		const source = connect[connect.length - 1];
		const { x, y } = $nodes.find((n) => n.id === source)!.position;
		const newNode = type2node(type, index, [x, y]);
		nodes.update((n) => [...n, newNode]);
		return [source, newNode.id];
	};

	const addNodeEdge = (type: Operator) => {
		const [source, target] = addNode(type);
		edges.update((e) => [
			...e,
			{
				id: `${source}-${target}`,
				source: source,
				target: target
			}
		]);
	};

	nodes.subscribe(() => {
		connect = generateNetwork($edges);
		pyCode = generateNetworkCode4Conn(connect, $nodes);
	});

	$inspect(selected).with(() => console.log(selected));
</script>

<!--
👇 By default, the Svelte Flow container has a height of 100%.
This means that the parent container needs a height to render the flow.
-->
<div class="relative h-full w-full overflow-hidden">
	<div class="absolute left-4 top-4 z-10 flex flex-col gap-4">
		<ControlLib click={addNodeEdge} />
		<Button size="lg" onclick={() => (showCode = true)}>Show Code</Button>
	</div>

	{#if selected && selected.data.type}
		<div transition:fly={{ x: 100, y: 0, duration: 300 }} class="absolute right-4 top-4 z-10">
			<ControlEditor
				type={selected.data.type as any}
				params={[selected.data.func, selected.data.params]}
				apply={(func, fparams) => apply(selected!.id, func, fparams)}
			/>
		</div>
	{/if}

	<SvelteFlow
		{nodes}
		{edges}
		colorMode={$mode}
		snapGrid={[20, 20]}
		defaultEdgeOptions={edgeOpt}
		fitView
		on:paneclick={() => (selected = null)}
		on:edgeclick={() => (selected = null)}
		on:nodeclick={(event) => (selected = event.detail.node as MyNode)}
	>
		<Controls position="bottom-right" />
		<Background variant={BackgroundVariant.Dots} />
	</SvelteFlow>
	{#if showCode}
		<div
			transition:fade
			class="absolute inset-0 z-20 flex flex-col items-center justify-center bg-white/30 backdrop-blur-md"
		>
			<Code lang="py" content={pyCode} class="h-4/5 w-3/4 text-sm" />
			<Button size="lg" onclick={() => (showCode = false)}>Close</Button>
		</div>
	{/if}
</div>
