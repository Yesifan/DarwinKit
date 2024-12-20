<script lang="ts">
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { gotoWithi18n } from '$lib/i18n';
	import Select from '$lib/components/select.svelte';
	import * as Alert from '$lib/components/ui/alert';
	import ConfigForm from '$lib/components/config-form.svelte';
	import Button, { buttonVariants } from '$lib/components/ui/button/button.svelte';
	import Code from '$lib/components/code.svelte';
	import * as m from '$lib/paraglide/messages';
	import { get, post } from '../api';
	import * as Dialog from '$lib/components/ui/dialog';
	import Input from '$lib/components/input.svelte';

	let modelType = $state('');
	let modelsOptions = $state<{ [key: string]: { model: any; trainer: any } } | null>(null);

	let dataset = $state('');
	let tokenizer = $state('');
	let datasetList = $state<string[]>([]);
	let tokenizerList = $state<string[]>([]);

	let modelKeys = $derived(
		modelsOptions ? Object.keys(modelsOptions).map((item) => ({ value: item, label: item })) : []
	);
	let modelOption = $state<any>({});
	let trainerOption = $state<any>({});
	let modelConfig = $derived(modelsOptions ? modelsOptions[modelType]?.model : null);
	let trainerConfig = $derived(modelsOptions ? modelsOptions[modelType]?.trainer : null);

	let forkName = $state<string | null>(null);

	let command = $derived.by(() => {
		const baseCommand = `darkit lm train --tokenizer ${tokenizer} --dataset ${dataset} ${modelType}`;
		return `${baseCommand} ${generateCommand(modelOption)} ${generateCommand(trainerOption)}`;
	});

	async function startTrain(command: string) {
		try {
			const res = await post(`/model/${modelType}/train`, {
				command: command
			});
			const data = await res.json();
			if (res.status !== 200) {
				toast.error(data.detail);
				return;
			}
			gotoWithi18n(`/lm/visual/${trainerOption.name}`);
		} catch (e) {
			console.error(e);
		}
	}

	async function createForkNetwork() {
		try {
			const res = await post(`/edit/init/${forkName}`, {
				model: modelType,
				m_conf: modelOption,
				t_conf: trainerOption
			});
			const data = await res.json();
			if (res.status !== 200) {
				toast.error(data.detail);
				return;
			}

			gotoWithi18n(`/lm/fork/${forkName}`);
		} catch (e) {
			console.error(e);
		}
	}

	// Generate command
	function generateCommand(config: any) {
		return Object.entries(config)
			.filter(([key, val]) => {
				return val !== null && val !== '' && val !== undefined;
			})
			.map(([key, val]) => `--${key} ${val}`)
			.join(' ');
	}

	onMount(async () => {
		const [modelsOptionsJson, resources] = await Promise.all([
			get('/models/options').then((res) => res.json()),
			get('/train/resources').then((res) => res.json())
		]);

		modelsOptions = modelsOptionsJson;
		if (modelsOptions) {
			if (modelType === '' || !Object.keys(modelsOptions).includes(modelType)) {
				modelType = Object.keys(modelsOptions)[0];
			}
		}
		if (resources) {
			[datasetList, tokenizerList] = resources;
			if (dataset === '' || !datasetList.includes(dataset)) {
				dataset = datasetList ? datasetList[0] : '';
			}
			if (tokenizer === '' || !tokenizerList.includes(tokenizer)) {
				tokenizer = tokenizerList ? tokenizerList[0] : '';
			}
		}
	});
</script>

<div class="h-full flex-1 overflow-y-auto overflow-x-hidden p-8">
	{#if modelsOptions}
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
					options={tokenizerList.map((v) => ({ value: v, label: v }))}
					tooltip={m.tokenizersTooltip()}
				/>
			</div>
			<ConfigForm key={modelType} config={modelConfig} bind:option={modelOption} />
		</section>
		<section class="py-4">
			<h3 class="mb-2 font-bold">Trainer Config</h3>

			<Select
				label={m.dataset()}
				class="mb-4 w-96"
				bind:value={dataset}
				options={datasetList.map((v) => ({ value: v, label: v }))}
			/>

			<ConfigForm key={modelType} config={trainerConfig} bind:option={trainerOption} />
		</section>

		<h3 class="mb-2 mt-8 scroll-m-20 text-2xl font-semibold tracking-tight">
			{m.generatedCommand()}
		</h3>

		<Code class="mt-8 max-h-96" lang="bash" content={command} wrap />

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
							bind:value={forkName}
						/>
					</div>
					<Dialog.Footer>
						<Button type="submit" onclick={createForkNetwork}>Create fork</Button>
					</Dialog.Footer>
				</Dialog.Content>
			</Dialog.Root>
			<Button size="lg" class="mt-8" onclick={() => startTrain(command)}>
				{m.startTrain()}
			</Button>
		</div>
	{:else}
		<div>Loading...</div>
	{/if}
</div>
