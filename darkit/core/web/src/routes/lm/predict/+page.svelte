<script lang="ts">
	import { untrack } from 'svelte';
	import { toast } from 'svelte-sonner';
	import Input from '$lib/components/input.svelte';
	import * as Alert from '$lib/components/ui/alert';
	import Select from '$lib/components/select.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import * as m from '$lib/paraglide/messages';
	import { post } from '../api';

	let { data } = $props();

	let name = $state('Example:complete');
	let prompt = $state('Hello');
	let ctx_len = $state(64);
	let predictText = $state('');

	let modelOptions = $derived(
		data.networks.map((model) => ({
			label: model.config.fork ? `[${model.config.fork}] ${model.name}` : model.name,
			value: model.name
		}))
	);

	$effect(() => {
		const defaultName = data.networks[0].name;
		untrack(() => {
			name = defaultName;
		});
	});

	const startPredict = async () => {
		predictText = prompt;
		try {
			const response = await post(`/model/${name}/predict`, { prompt, ctx_len });

			if (response.status === 200) {
				const reader = response.body?.getReader();
				const decoder = new TextDecoder();

				while (reader) {
					const { done, value } = await reader?.read();
					if (done) {
						break;
					}
					predictText += decoder.decode(value);
				}
			} else {
				const data = await response.json();
				toast.error(data.detail);
			}
		} catch (error) {
			console.error('Error:', error);
		}
	};
</script>

<div class="h-full flex-1 overflow-x-hidden p-8">
	<Alert.Root class="text-primary/60 [&:has(svg)]:pl-4">
		<Alert.Description>
			<span class="text-xl"> 🍾 </span>
			{m.badageAlert({ icon: '[?]' })}
		</Alert.Description>
	</Alert.Root>
	<div class="grid grid-cols-4 gap-4 py-8">
		<Select
			label={m.modelName()}
			bind:value={name}
			options={modelOptions}
			tooltip={m.modelNameTooltip()}
		/>
		<Input label="ctx_len" bind:value={ctx_len} type="number" tooltip={m.cxtLenTooltip()} />
		<Input label={m.prompt()} bind:value={prompt} class="col-span-4" tooltip={m.promptTooltip()} />
	</div>
	<div class="bg-gray min-h-48 rounded-md bg-gray-950 p-4 font-mono text-white dark:bg-gray-800">
		{predictText}
	</div>
	<div class="w-full pb-4 text-right">
		<Button size="lg" class="mt-8" onclick={startPredict}>
			{m.startPredict()}
		</Button>
	</div>
</div>
