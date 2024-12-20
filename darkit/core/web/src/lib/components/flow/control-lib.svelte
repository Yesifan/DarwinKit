<script lang="ts">
	import { OperatorDetail } from '$lib/flow';
	import { cn } from '$lib/utils';
	import { Button } from '../ui/button';
	import type { HTMLAttributes } from 'svelte/elements';
	import type { Operator } from '$lib/flow';

	interface Props extends HTMLAttributes<HTMLElement> {
		click?: (name: Operator) => void;
	}

	let { class: className, click, ...restProps }: Props = $props();
</script>

<section
	class={cn(
		'bg-background/60 border-border flex flex-col gap-4 rounded border p-4 shadow',
		className
	)}
	{...restProps}
>
	<h4 class="p-1 text-sm">Operators</h4>
	<div class="flex flex-col gap-2">
		{#each Object.keys(OperatorDetail) as operator}
			{#if operator !== 'Input'}
				<Button
					variant="outline"
					class="border-primary"
					onclick={() => click?.(operator as Operator)}
				>
					{operator}
				</Button>
			{/if}
		{/each}
	</div>
</section>
