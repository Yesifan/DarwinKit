<script lang="ts" module>
	import type { WithElementRef } from 'bits-ui';
	import type { HTMLAnchorAttributes, HTMLInputAttributes } from 'svelte/elements';
	import { type VariantProps, tv } from 'tailwind-variants';

	export const inputVariants = tv({
		base: 'flex w-full max-w-sm gap-1.5',
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

	export type InputVariant = VariantProps<typeof inputVariants>['variant'];
	type Poros = { label?: string; tooltip?: string; variant?: InputVariant };
	export type InputProps = WithElementRef<HTMLInputAttributes> &
		WithElementRef<HTMLAnchorAttributes> &
		Poros;
</script>

<script lang="ts">
	import { Input } from './ui/input';
	import { Label } from '$lib/components/ui/label/index.js';
	import HelperTooltip from './helper-tooltip.svelte';
	import { cn } from '$lib/utils';

	let {
		ref = $bindable(null),
		value = $bindable(''),
		label,
		tooltip,
		class: className,
		variant = 'default',
		href = undefined,
		type,
		children,
		...restProps
	}: InputProps = $props();

	// 初始值处理
	if (type === 'number' && value && Number(value) < 0.001 && Number(value) !== 0) {
		value = Number(value).toExponential();
	}

	// type === 'number' 时, 在用户输入结束的时候，如果输入的值小于0.001 自动改写为科学计数法。
	function handleInput() {
		if (type === 'number' && value && Number(value) < 0.001 && Number(value) !== 0) {
			value = Number(value).toExponential();
		}
	}
</script>

<div class={cn(inputVariants({ variant, className }))}>
	{#if tooltip}
		<HelperTooltip for={label} {tooltip} class="flex-shrink-0">
			{label}
		</HelperTooltip>
	{:else}
		<Label class="flex-shrink-0" for={label}>{label}</Label>
	{/if}
	<Input id={label} {type} bind:value onblur={handleInput} {...restProps} />
</div>
