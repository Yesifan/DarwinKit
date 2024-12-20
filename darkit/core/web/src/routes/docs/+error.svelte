<script>
	import Button from '$lib/components/ui/button/button.svelte';
	import * as m from '$lib/paraglide/messages';
</script>

<div class="not-prose flex h-full w-full flex-col items-center justify-center py-8 text-center">
	<h1 class="text-9xl font-bold">404</h1>
	<p class="my-4 text-2xl">{m.not_found_docs_tip()}</p>
	<Button href="/" class="mt-8" size="lg">{m.back_home()}</Button>
</div>
