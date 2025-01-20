<script>
	import Navbar from '$lib/components/navbar.svelte';
	import * as m from '$lib/paraglide/messages';

	const contents = [
		{
			caption: m.toolkit(),
			contents: [[m.h_network(), '/toolkit/flow']]
		}
	];
</script>

<Navbar {contents} />
<slot />
