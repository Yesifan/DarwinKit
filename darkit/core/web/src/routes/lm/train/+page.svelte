<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { gotoWithi18n } from '$lib/i18n';
	import * as Alert from '$lib/components/ui/alert';
	import Select from '$lib/components/select.svelte';
	import ConfigForm from '$lib/components/config-form.svelte';
	import { Button, buttonVariants } from '$lib/components/ui/button';
	import Code from '$lib/components/code.svelte';
	import * as m from '$lib/paraglide/messages';
	import {
		createForkNetwork as createForkNetworkApi,
		getTrainCommand,
		startTrainModel
	} from '$lib/apis/lm';
	import * as Dialog from '$lib/components/ui/dialog';
	import Input from '$lib/components/input.svelte';

	let { data } = $props();

	let modelType = $state(Object.keys(data.modelsOptions)[0]);

	let dataset = $state(data.resources[0][0]);
	let tokenizer = $state(data.resources[1][0]);

	let fork = $state<string | null>(null);
	let resume = $state('');

	let modelOption = $state<any>({});
	let trainerOption = $state<any>({});
	let modelKeys = $derived(
		data.modelsOptions
			? Object.keys(data.modelsOptions).map((item) => ({ value: item, label: item }))
			: []
	);
	let resumeList = $derived(
		data.trainedNetworks ? data.trainedNetworks.map((v) => ({ value: v.name, label: v.name })) : []
	);
	let datasetList = $derived(data.resources[0].map((item) => ({ value: item, label: item })));
	let tokenizerList = $derived(data.resources[1].map((item) => ({ value: item, label: item })));
	let modelConfig = $derived(data.modelsOptions ? data.modelsOptions[modelType]?.model : null);
	let trainerConfig = $derived(data.modelsOptions ? data.modelsOptions[modelType]?.trainer : null);

	let command = $derived.by(async () => {
		return getTrainCommand(modelType, fork, resume, dataset, tokenizer, modelOption, trainerOption);
	});

	async function startTrain() {
		try {
			const type = modelType;
			await startTrainModel(type, fork, resume, dataset, tokenizer, modelOption, trainerOption);
			gotoWithi18n(`/lm/visual/${trainerOption.name}`);
		} catch (e) {
			console.error(e);
			toast.error('Failed to start training');
		}
	}

	async function createForkNetwork() {
		if (fork) {
			try {
				await createForkNetworkApi(fork, modelType, modelOption, trainerOption);
				gotoWithi18n(`/lm/fork/${fork}`);
			} catch (e) {
				console.error(e);
				toast.error('Failed to create fork network');
			}
		}
	}
</script>

<div class="h-full flex-1 overflow-y-auto overflow-x-hidden p-8">
	{#if data.modelsOptions}
		<Alert.Root class="text-primary/60 [&:has(svg)]:pl-4">
			<Alert.Description>
				<span class="text-xl"> 🍾 </span>
				{m.badageAlert({ icon: '[?]' })}
			</Alert.Description>
		</Alert.Root>
		<section class="py-4">
			<h3 class="mb-2 font-bold">Model Config</h3>
			<div class="flex gap-4 pb-4">
				<Select class="flex-1" label={m.modelType()} bind:value={modelType} options={modelKeys} />
				<Select
					class="flex-1"
					label="Tokenizers"
					bind:value={tokenizer}
					options={tokenizerList}
					tooltip={m.tokenizersTooltip()}
				/>
				<Select
					class="flex-1"
					label="Resume"
					bind:value={resume}
					options={resumeList}
					tooltip={m.resumeTooltip()}
				/>
			</div>
			<ConfigForm key={modelType} config={modelConfig} bind:option={modelOption} />
		</section>
		<section class="py-4">
			<h3 class="mb-2 font-bold">Trainer Config</h3>
			<Select label={m.dataset()} class="mb-4 w-96" bind:value={dataset} options={datasetList} />
			<ConfigForm key={modelType} config={trainerConfig} bind:option={trainerOption} />
		</section>

		<h3 class="mb-2 mt-8 scroll-m-20 text-2xl font-semibold tracking-tight">
			{m.generatedCommand()}
		</h3>

		{#await command}
			<Code class="mt-8 max-h-96" lang="bash" content="Updating..." wrap />
		{:then command}
			<Code class="mt-8 max-h-96" lang="bash" content={command} wrap />
		{:catch error}
			<Code class="mt-8 max-h-96" lang="bash" content={error.detail} wrap />
		{/await}

		<div class="w-full text-right">
			<Dialog.Root>
				<Dialog.Trigger class={buttonVariants({ variant: 'outline', size: 'lg' })}>
					{m.fork()}
				</Dialog.Trigger>
				<Dialog.Content>
					<Dialog.Header>
						<Dialog.Title>Create a new network fork</Dialog.Title>
						<Dialog.Description>
							A fork is a copy of a network. You can edit the new network and train it.
						</Dialog.Description>
					</Dialog.Header>
					<div>
						<Input
							label="New network name"
							placeholder="Enter a name for the fork"
							class="w-full"
							variant="row"
							bind:value={fork}
						/>
					</div>
					<Dialog.Footer>
						<Button type="submit" onclick={createForkNetwork}>Create fork</Button>
					</Dialog.Footer>
				</Dialog.Content>
			</Dialog.Root>
			<Button size="lg" class="mt-8" onclick={startTrain}>
				{m.startTrain()}
			</Button>
		</div>
	{:else}
		<div>Loading...</div>
	{/if}
</div>
