<script lang="ts">
	import type { PageData } from './$types.js';
	import UploadForm from './upload-form.svelte';
	import type { NXEdge, NXNode } from '$lib/components/model-editor.svelte';
	import ModelEditor from '$lib/components/model-editor.svelte';
	import ControlForm from './control-form.svelte';
	import * as m from '$lib/paraglide/messages';
	import type { Node } from '@xyflow/svelte';
	import type { Infer } from 'sveltekit-superforms';
	import type { ControlFormSchema } from './schema.js';
	import TrainForm from './train-form.svelte';

	let { data }: { data: PageData } = $props();
	let selected: Node | null = $state(null);

	let cname = $state('');
	let graph: [NXNode[], NXEdge[]] | null = $state(null);
	let sconf: Record<string, any> = $state({});

	let compositeNodes: NXNode[] | undefined = $derived.by(() => {
		return graph?.[0].map(([nid, { label, ...rest }]) => {
			if (nid in sconf) {
				const updated = sconf[nid];

				const nlabel = `${label ?? nid}(${updated.type})`;
				return [nid, { label: nlabel, ...rest }];
			}
			return [nid, { label, ...rest }];
		});
	});

	let updateGraph = (id: string, formdata: Infer<ControlFormSchema>) => {
		if (selected && graph) {
			sconf[id] = formdata;
		}
	};

	$inspect(sconf);
</script>

<section class="flex size-full flex-col">
	<h2
		class="scroll-m-20 px-8 py-4 text-3xl font-semibold tracking-tight transition-colors first:mt-0"
	>
		{m.spiking_title()}
	</h2>
	{#if graph && compositeNodes}
		<div class="relative flex-1">
			<ModelEditor class="size-full" bind:selected nodes={compositeNodes} edges={graph[1]} />
			{#if selected}
				<ControlForm
					data={data.controlForm}
					id={selected.id}
					onupdate={updateGraph}
					class="absolute right-12 top-4 w-96"
				/>
			{/if}
			<TrainForm data={data.trainForm} {cname} {sconf} class="absolute bottom-4 right-12" />
		</div>
	{:else}
		<div class="flex flex-col items-center">
			<h1 class="py-1 text-2xl font-semibold tracking-tight">{m.spiking_upload()}</h1>
			<p class="text-muted-foreground pl-6">{m.spiking_desc()}</p>
			<UploadForm data={data.uploadForm} class="w-96 py-8" bind:graph bind:cname />
		</div>
	{/if}
</section>
