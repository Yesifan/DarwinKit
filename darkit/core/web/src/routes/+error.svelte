<script>
	import { page } from '$app/stores';
	import Button from '$lib/components/ui/button/button.svelte';
	import * as m from '$lib/paraglide/messages';
</script>

<div class="flex h-full w-full flex-col items-center justify-center text-center">
	<h1 class="text-9xl font-bold">{$page.status}</h1>
	<p class="my-4 text-2xl">{$page.error?.message ?? m.not_found_tip()}</p>
	<Button href="/" class="mt-8" size="lg">{m.back_home()}</Button>
</div>
