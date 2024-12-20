<script>
	import Navbar from '$lib/components/navbar.svelte';
	import * as m from '$lib/paraglide/messages';

	const contents = [
		{
			caption: m.run(),
			contents: [
				[m.train(), '/lm/train'],
				[m.fork(), '/lm/fork'],
				[m.predict(), '/lm/predict'],
				[m.upload(), '/lm/upload'],
				[m.visualize(), '/lm/visual']
			]
		}
	];
</script>

<Navbar {contents} />
<slot />
