<script lang="ts">
	import { Checkbox as CheckboxPrimitive } from 'bits-ui';
	import Check from 'lucide-svelte/icons/check';
	import Minus from 'lucide-svelte/icons/minus';
	import { cn } from '$lib/utils.js';
	import Button from './ui/button/button.svelte';

	type $$Props = CheckboxPrimitive.Props;
	type $$Events = CheckboxPrimitive.Events;

	let className: $$Props['class'] = undefined;
	export let checked: $$Props['checked'] = false;
	export { className as class };
</script>

<Button class={cn(checked ? 'bg-primary/10' : '', className)} {...$$props}>
	<slot />
</Button>
