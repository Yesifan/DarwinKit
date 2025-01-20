<script>
	import Navbar from '$lib/components/navbar.svelte';
	import * as m from '$lib/paraglide/messages';

	const contents = [
		{
			caption: m.toolkit(),
			contents: [
				[m.t_spiking(), '/toolkit/spiking'],
				[m.t_network(), '/toolkit/flow'],
				[m.visualTrain(), '/toolkit/visual']
			]
		}
	];
</script>

<Navbar {contents} />
<slot />
