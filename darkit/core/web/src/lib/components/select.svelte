<script lang="ts" module>
	import type { SelectSingleRootProps } from 'bits-ui';
	import { type VariantProps, tv } from 'tailwind-variants';

	type Option = { value: string; label: string; disabled?: boolean };
	type Poros = {
		options: Option[];
		placeholder?: string;
		disabled?: boolean | undefined;
		tooltip?: string;
		label: string;
		class?: string;
	};

	export const SelectVariants = tv({
		base: 'flex w-full max-w-sm  gap-1.5',
		variants: {
			variant: {
				default: 'flex-col',
				row: 'flex-row items-center gap-2'
			}
		},
		defaultVariants: {
			variant: 'default'
		}
	});

	export type SelectVariant = VariantProps<typeof SelectVariants>['variant'];

	export type SelectProps = Omit<SelectSingleRootProps, 'type'> &
		Poros & {
			variant?: SelectVariant;
		};
</script>

<script lang="ts">
	import { cn } from '$lib/utils';
	import { Label } from '$lib/components/ui/label/index.js';
	import * as Select from '$lib/components/ui/select/index.js';
	import HelperTooltip from './helper-tooltip.svelte';

	let {
		value = $bindable(),
		label,
		options,
		tooltip,
		placeholder,
		class: className,
		variant = 'default',
		children,
		...restProps
	}: SelectProps = $props();

	let triggerVal = $derived(options.find((item) => item.value === value)?.label);
</script>

<div class={cn(SelectVariants({ variant, className }))}>
	{#if tooltip}
		<HelperTooltip for={label} {tooltip}>
			{label}
		</HelperTooltip>
	{:else}
		<Label for={label}>{label}</Label>
	{/if}

	<Select.Root type="single" bind:value {...restProps}>
		<Select.Trigger>
			{triggerVal ?? placeholder}
		</Select.Trigger>
		<Select.Content id={label}>
			{#each options as item}
				<Select.Item value={item.value} label={item.label} disabled={item.disabled}>
					{item.label}
				</Select.Item>
			{/each}
		</Select.Content>
	</Select.Root>
</div>
