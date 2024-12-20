<script lang="ts">
	import { page } from '$app/stores';
	import * as m from '$lib/paraglide/messages';
	import * as Sidebar from '$lib/components/ui/sidebar';

	type Props = { contents?: { caption: string; contents: string[][] }[] };

	let {
		contents = [
			{
				caption: m.run(),
				contents: [
					[m.train(), '/toolkits/train'],
					[m.predict(), '/toolkits/predict']
				]
			},
			{
				caption: m.visualize(),
				contents: [[m.model(), '/toolkits/visual']]
			}
		]
	}: Props = $props();

	let pathname = $derived($page.url.pathname);
</script>

<Sidebar.Root variant="floating">
	<Sidebar.Content>
		{#each contents as item (item.caption)}
			<Sidebar.Group>
				<Sidebar.GroupLabel>
					{item.caption}
				</Sidebar.GroupLabel>
				<Sidebar.GroupContent>
					<Sidebar.Menu>
						{#each item.contents as [title, url] (url)}
							<Sidebar.MenuItem>
								<Sidebar.MenuButton isActive={pathname.includes(url)}>
									{#snippet child({ props })}
										<a href={url} {...props}>
											<span>{title}</span>
										</a>
									{/snippet}
								</Sidebar.MenuButton>
							</Sidebar.MenuItem>
						{/each}
					</Sidebar.Menu>
				</Sidebar.GroupContent>
			</Sidebar.Group>
		{/each}
	</Sidebar.Content>
</Sidebar.Root>
