<script lang="ts">
	import { cn } from '$lib/utils';
	import type { ModelLabel } from '$lib/apis/lm';
	import Input from '$lib/components/input.svelte';
	import Select from './select.svelte';
	import Switch from './switch.svelte';

	type Props = {
		key: string;
		class?: string;
		disabled?: boolean;
		config: Record<string, ModelLabel>;
		option: { [key: string]: any };
	};

	let { key, class: className, config, disabled = false, option = $bindable() }: Props = $props();
	let prevKey = $state<string | undefined>();

	$effect(() => {
		if (key !== prevKey) {
			if (prevKey !== null) {
				Object.keys(option).forEach((key) => {
					delete option[key];
				});
			}

			Object.entries(config).forEach(([key, value]) => {
				if (value.default) {
					option[key] = option[key] ?? value.default;
				} else if (value.type === 'bool') {
					option[key] = option[key] ?? false;
				} else {
					if (value.type === 'int' || value.type === 'float') {
						option[key] = option[key] ?? 0;
					} else {
						option[key] = option[key] ?? '';
					}
				}
			});
			prevKey = key;
		}
	});
</script>

<div class={cn('grid grid-cols-4 gap-4', className)}>
	{#each Object.entries(config) as [key, value]}
		{#if value.type === 'str'}
			{#if option[key] !== undefined}
				<Input
					label={key}
					{disabled}
					bind:value={option[key]}
					tooltip={value.comment}
					required={value.required}
				/>
			{/if}
		{:else if value.type === 'int' || value.type === 'float'}
			{#if option[key] !== undefined}
				<Input
					label={key}
					type="number"
					{disabled}
					bind:value={option[key]}
					tooltip={value.comment}
					required={value.required}
				/>
			{/if}
		{:else if value.type === 'bool'}
			{#if typeof option[key] === 'boolean'}
				<Switch label={key} bind:checked={option[key]} tooltip={value.comment} {disabled} />
			{/if}
		{:else if Array.isArray(value.type)}
			<Select
				label={key}
				bind:value={option[key]}
				{disabled}
				options={value.type.map((model) => ({ value: model, label: model }))}
				tooltip={value.comment}
			/>
		{/if}
	{/each}
</div>
