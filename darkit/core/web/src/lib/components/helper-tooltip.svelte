<script lang="ts">
	import { BadgeHelp } from 'lucide-svelte';
	import { Label } from '$lib/components/ui/label/index.js';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { Label as LabelPrimitive } from 'bits-ui';

	let {
		tooltip,
		ref = $bindable(null),
		class: className,
		children,
		...restProps
	}: LabelPrimitive.RootProps & { tooltip?: string } = $props();
</script>

<Tooltip.Provider>
	<Tooltip.Root>
		<Label {...restProps} class="whitespace-nowrap">
			<Tooltip.Trigger>
				{@render children?.()}
				<BadgeHelp size={14} class="inline-block" />
			</Tooltip.Trigger>
		</Label>
		<Tooltip.Content>
			<p>{tooltip}</p>
		</Tooltip.Content>
	</Tooltip.Root>
</Tooltip.Provider>
