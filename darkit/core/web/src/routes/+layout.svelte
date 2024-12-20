<script>
	import '../app.css';
	import '../styles/prism-one-dark.css';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import { ParaglideJS } from '@inlang/paraglide-sveltekit';
	import { i18n } from '$lib/i18n';
	import Header from '$lib/components/header.svelte';
	import { Toaster } from '$lib/components/ui/sonner';
	import { ModeWatcher } from 'mode-watcher';
</script>

<ParaglideJS {i18n}>
	<Sidebar.Provider class="app flex h-screen flex-col">
		<ModeWatcher />
		<Header />
		<main class="flex h-[calc(100vh-3.5rem)]">
			<slot />
		</main>
		<Toaster />
	</Sidebar.Provider>
</ParaglideJS>
