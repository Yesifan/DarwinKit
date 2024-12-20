<script lang="ts">
	import { page } from '$app/stores';
	import * as m from '$lib/paraglide/messages';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import * as Collapsible from './ui/collapsible';
	import { ChevronDown } from 'lucide-svelte';

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
		<Sidebar.Group>
			<Sidebar.GroupContent>
				<Sidebar.Menu>
					{#each contents as item (item.caption)}
						<Collapsible.Root open class="group/collapsible">
							<Sidebar.MenuItem>
								<Collapsible.Trigger>
									{#snippet child({ props })}
										<Sidebar.MenuButton {...props}>
											{item.caption}
											<ChevronDown
												class="ml-auto transition-transform group-data-[state=open]/collapsible:rotate-180"
											/>
										</Sidebar.MenuButton>
									{/snippet}
								</Collapsible.Trigger>

								<Collapsible.Content>
									<Sidebar.MenuSub>
										{#each item.contents as [title, url] (url)}
											<Sidebar.MenuSubItem>
												<Sidebar.MenuSubButton isActive={pathname.includes(url)}>
													{#snippet child({ props })}
														<a href={url} {...props}>
															<span>{title}</span>
														</a>
													{/snippet}
												</Sidebar.MenuSubButton>
											</Sidebar.MenuSubItem>
										{/each}
									</Sidebar.MenuSub>
								</Collapsible.Content>
							</Sidebar.MenuItem>
						</Collapsible.Root>
					{/each}
				</Sidebar.Menu>
			</Sidebar.GroupContent>
		</Sidebar.Group>
	</Sidebar.Content>
</Sidebar.Root>
