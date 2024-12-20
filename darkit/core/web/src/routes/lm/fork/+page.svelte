<script lang="ts">
	import { untrack } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { post } from '$lib/api';
	import { gotoWithi18n } from '$lib/i18n.js';
	import Select from '$lib/components/select.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import * as Card from '$lib/components/ui/card';
	import ConfigForm from '$lib/components/config-form.svelte';
	import * as m from '$lib/paraglide/messages';

	let { data } = $props();

	let forkModel = $state('');
	let trainerOption = $state<any>({});

	let dataset = $state('');
	let tokenizer = $state('');

	let modelType = $derived(data.forkModels?.[0]?.model);
	let modelOption = $derived(data.forkModels.find((v) => v.name == forkModel)?.m_conf);
	// Fork Model 下拉框选项
	let forkModelOptions = $derived(
		data.forkModels.map((v) => ({ value: v.name, label: `[${v.model}] ${v.name}` }))
	);
	let datasetOptions = $derived(data.datasetList.map((v) => ({ value: v, label: v })));
	let tokenizerOptions = $derived(data.tokenizerList.map((v) => ({ value: v, label: v })));

	$effect(() => {
		const { forkModels, datasetList, tokenizerList } = data;
		untrack(() => {
			forkModel = forkModels[0].name;
			dataset = datasetList[0];
			tokenizer = tokenizerList[0];
		});
	});

	$effect(() => {
		const currentModel = data.forkModels.find((v) => v.name == forkModel);
		if (currentModel) {
			untrack(() => {
				trainerOption = currentModel.t_conf;
			});
		}
	});

	async function startTrain() {
		try {
			if (!forkModel) {
				toast.error('Please select fork model');
				return;
			} else if (!trainerOption.name) {
				toast.error('please enter trainer name');
				return;
			}
			const res = await post('/lm/v2/model/train/', {
				type: modelType,
				fork: forkModel,
				dataset: dataset,
				tokenizer: tokenizer,
				m_conf: modelOption,
				t_conf: trainerOption
			});
			if (res.status !== 200) {
				const data = await res.json();
				toast.error(data.detail);
				return;
			}
			gotoWithi18n(`/lm/visual/${trainerOption.name}`);
		} catch (e) {
			console.error(e);
		}
	}
</script>

<div class="h-full flex-1 overflow-y-auto overflow-x-hidden p-8">
	{#if modelType}
		<section class="py-4">
			<h3 class="mb-2 font-bold">Config</h3>
			<div class="grid grid-cols-2 gap-4">
				<Select label="Forked Model" bind:value={forkModel} options={forkModelOptions} />
			</div>
		</section>
		<Card.Root class="my-4">
			<Card.Header>
				<Card.Title>Trainer Config</Card.Title>
			</Card.Header>
			<Card.Content>
				<div class="grid grid-cols-2">
					<Select
						label="Tokenizers"
						bind:value={tokenizer}
						options={tokenizerOptions}
						tooltip={m.tokenizersTooltip()}
					/>
					<Select
						label={m.dataset()}
						class="mb-4 w-96"
						bind:value={dataset}
						options={datasetOptions}
					/>
				</div>
				<ConfigForm
					key={modelType}
					config={data.options[modelType].trainer}
					bind:option={trainerOption}
				/>
			</Card.Content>
		</Card.Root>

		{#if modelOption}
			<Card.Root class="my-4">
				<Card.Header>
					<Card.Title>Model Config</Card.Title>
				</Card.Header>
				<Card.Content class="grid grid-cols-4 gap-1 2xl:grid-cols-8">
					{#each Object.entries(modelOption) as [key, val]}
						<div>
							<div class="text-primary/60 text-sm">{key}</div>
							<div class="text-primary">{val}</div>
						</div>
					{/each}
				</Card.Content>
			</Card.Root>
		{/if}

		<div class="w-full text-right">
			<Button size="lg" variant="outline" href={`/lm/fork/${forkModel}`}>
				{m.visualEdit()}
			</Button>
			<Button size="lg" onclick={startTrain}>
				{m.startTrain()}
			</Button>
		</div>
	{:else}
		<div class="text-center text-2xl text-gray-400">NO Fork Model, Please Create First.</div>
	{/if}
</div>
