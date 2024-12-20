const BASE_API = '/api/lm';

export const get = (url: string, params?: any, opt?: RequestInit) => {
	const fullURL = new URL(BASE_API + url, location.origin);
	fullURL.search = new URLSearchParams(params).toString();
	return fetch(fullURL, opt);
};

export const post = (url: string, params?: any, opt?: RequestInit) => {
	const fullURL = new URL(BASE_API + url, location.origin);
	const body = params instanceof FormData ? params : JSON.stringify(params);
	const headers: any =
		params instanceof FormData
			? { ...opt?.headers }
			: { 'Content-Type': 'application/json', ...opt?.headers };
	return fetch(fullURL, {
		method: 'POST',
		...opt,
		body,
		headers
	});
};

export const delete_ = (url: string, params?: any, opt?: RequestInit) => {
	return post(url, params, { ...opt, method: 'DELETE' });
};
