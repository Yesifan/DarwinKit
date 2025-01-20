<script lang="ts" module>
	import type {
		NXEdge,
		NXNode,
		Props as ModelEditorProps
	} from '$lib/components/model-editor.svelte';
	import { get, post } from '$lib/api';

	type ModuleFuncs = { name: string; body: string };

	const releaseNetwork = async (name: string) => {
		const response = await post('/lm/edit/release', { name: name });
		if (response.status === 200) {
			toast.success('Release success');
			window.history.back();
		} else {
			toast.error('Release failed');
		}
	};
</script>

<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { SaveIcon } from 'lucide-svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import CodeEditor from '$lib/components/code-editor.svelte';
	import ModelEditor from '$lib/components/model-editor.svelte';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import * as m from '$lib/paraglide/messages';
	import { untrack } from 'svelte';
	import { Skeleton } from '$lib/components/ui/skeleton/index.js';

	let { data } = $props();
	let nodes = $state<NXNode[]>([]);
	let edges = $state<NXEdge[]>([]);
	let level = $state(0);
	let selected: string | null = $state(null);
	let currentModule = $state<ModuleFuncs[]>([]);
	let moduleFuncMap = $state(new Map<string, ModuleFuncs[]>());

	$effect(() => {
		data.res.then((data) => {
			untrack(() => {
				if ('nodes' in data && data.nodes && data.edges) {
					nodes = data.nodes;
					edges = data.edges;
				}
			});
		});
	});

	const onNodeClick: ModelEditorProps['onNodeClick'] = async (event) => {
		const { node } = event.detail;
		selected = node.id;
		if (!moduleFuncMap.has(node.id)) {
			const response = await get(`/lm/edit/source`, { id: node.id });
			if (response.status === 200) {
				const funcs: ModuleFuncs[] = await response.json();
				moduleFuncMap.set(node.id, funcs);
				currentModule = funcs;
			}
		} else {
			currentModule = moduleFuncMap.get(node.id)!;
		}
	};

	const onNodeExpand = async (id: string) => {
		if (nodes && edges) {
			const nodeList = nodes.map((item) => item[0]);
			const response = await post('/lm/edit/subgraph', { id, nodes: nodeList });
			if (response.status === 200) {
				level += 1;
				const res: { nodes: NXNode[]; edges: NXEdge[] } = await response.json();
				const nodesWithLevel: NXNode[] = res.nodes.map(([id, data]) => [id, { ...data, level }]);
				nodes = [...nodesWithLevel, ...nodes.filter(([sid]) => sid !== id)];
				edges = [...res.edges, ...edges.filter(([sid, tid]) => !(sid == id || tid == id))];
			}
		}
	};

	const onCodeChange = async (id: string, code: string) => {
		currentModule = currentModule.map((item) => {
			if (item.name === id) {
				return { name: id, body: code };
			}
			return item;
		});
		if (selected) {
			moduleFuncMap.set(selected, currentModule);
		}
	};

	const onCommit = async (name: string) => {
		if (selected && moduleFuncMap.keys().toArray().length > 0) {
			const code = moduleFuncMap.get(selected)?.find((item) => item.name === name)?.body;
			const response = await post('/lm/edit/commit', {
				module: selected,
				name,
				code
			});
			if (response.status === 200) {
				toast.success('Commit success');
			} else {
				toast.error('Commit failed');
			}
		}
	};
</script>

{#await data.res}
	<div class="w-full">
		<div class="grid h-full w-full grid-cols-10 gap-4 py-4 pr-2">
			<Skeleton class="col-span-6" />
			<div class="col-span-4 flex h-full flex-col gap-4 overflow-hidden">
				<div class="flex items-center justify-between">
					<span class="text-lg font-semibold">{data.name}</span>
				</div>
				<Skeleton class="h-full" />
			</div>
		</div>
	</div>
{:then res}
	<div class="grid h-full w-full grid-cols-10 gap-4 py-4 pr-2">
		{#if nodes && edges}
			<ModelEditor class="col-span-6" {nodes} {edges} {onNodeExpand} {onNodeClick} />
		{/if}
		<div class="col-span-4 flex h-full flex-col gap-4 overflow-hidden">
			<div class="flex items-center justify-between">
				<span class="text-lg font-semibold">{data.name}</span>
				<Button variant="destructive" size="lg" onclick={() => releaseNetwork(data.name)}>
					Release
				</Button>
			</div>
			{#if currentModule.length === 0}
				<div class="flex h-full items-center justify-center text-gray-400">EMPTY</div>
			{:else}
				<ScrollArea class="overflow-y h-full " orientation="vertical">
					<div class="flex flex-col gap-2">
						{#each currentModule as { name, body }}
							<div class="relative">
								<CodeEditor code={body} onChange={(e) => onCodeChange(name, e.detail)} />
								<Button
									class="absolute right-2 top-2 size-6"
									size="icon"
									variant="outline"
									onclick={() => onCommit(name)}
								>
									<SaveIcon class="size-4" />
								</Button>
							</div>
						{/each}
					</div>
				</ScrollArea>
			{/if}
		</div>
	</div>

	{#if 'code' in res}
		<AlertDialog.Root open={res.code === 1}>
			<AlertDialog.Content>
				<AlertDialog.Header>
					<AlertDialog.Title>Aleady loaded {res.name} model.</AlertDialog.Title>
					<AlertDialog.Description>
						Do you want to edit or release the {res.name} model? Or back to fork page.
					</AlertDialog.Description>
				</AlertDialog.Header>
				<AlertDialog.Footer>
					<Button variant="secondary" href="/lm/fork">{m.back()}</Button>
					<Button href={`/lm/fork/${res.name}`}>
						Check {res.name} Model
					</Button>
				</AlertDialog.Footer>
			</AlertDialog.Content>
		</AlertDialog.Root>
	{/if}
{/await}
