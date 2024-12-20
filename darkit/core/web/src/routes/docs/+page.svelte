<script>
	import { gotoWithi18n } from '$lib/i18n';
	import Navigation from './navigation.json';

	const firstDocPath = Navigation[0].children[0].path;
	gotoWithi18n(firstDocPath), { replaceState: true };
	console.log('firstDocPath', firstDocPath);
</script>

<h1>Docs</h1>
