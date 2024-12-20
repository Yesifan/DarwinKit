<script lang="ts">
	import { cn } from '$lib/utils';
	import { Label } from '$lib/components/ui/label/index.js';
	import HelperTooltip from './helper-tooltip.svelte';
	import Switch from './ui/switch/switch.svelte';

	import type { SwitchRootProps } from 'bits-ui';

	type Props = { label: string | undefined; tooltip: string | undefined };

	let {
		class: className,
		tooltip,
		label,
		checked = $bindable(false),
		...restProps
	}: SwitchRootProps & Props = $props();
</script>

<div class={cn('flex w-full max-w-sm flex-col gap-1.5', className)}>
	{#if tooltip}
		<HelperTooltip for={label} {tooltip}>
			{label}
		</HelperTooltip>
	{:else}
		<Label for={label}>{label}</Label>
	{/if}
	<div class="flex h-10 items-center">
		<Switch id={label} bind:checked {...restProps} />
	</div>
</div>
