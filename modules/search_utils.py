import requests
from typing import Literal, Optional
import re
import wikipediaapi

from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

# User-Agent
# 相似度模型计算 query 和 网页中每一段的相似度，筛选相关度高的。

default_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:92.0) Gecko/20100101 Firefox/92.0",
}
search_url = {
    "baidu": "https://www.baidu.com/s?wd=",
    "wikipedia": "https://en.wikipedia.org/wiki/Special:Search?search=",
    "bing": "https://www.bing.com/search?q="
}
SEARCH_ENGINES = Literal["baidu", "duckduckgo", "wikipedia", "bing"]


def web_search(
    search: str,
    search_engine: SEARCH_ENGINES = "bing",
    search_site: str | None = None,
    max_results: int = 2,
) -> list:
    """
    get web search results, and return a list of dictionaries, each consists of title, href and body
    :param search: query
    :param search_engine: 'baidu' and 'duckduckgo' is available
    :param search_site: specific the site you want to search on
    :param max_results: mark the maximum number of web search results.
    :return: list of dictionaries, [{'title', 'href', 'body'}]
    """
    if search_engine == "baidu":
        result = baidu_search(
            search=search, search_site=search_site, search_max_results=max_results
        )
    elif search_engine == "duckduckgo":
        result = ddg_search(
            search=search, search_site=search_site, search_max_results=max_results
        )
    elif search_engine == "wikipedia":
        result = wikipedia_search(search=search, search_max_results=max_results)
    elif search_engine == "bing":
        result = bing_search(
            search=search, search_max_results=max_results
        )
    return result

def url_response_parse(
    url: str,
    header = default_headers,
    max_result_length: int = 1000,
) -> Optional[str]:
    """
    parse the html page from input url, and get a long string of page content.
    :param url: string of html page.
    :param headers: User-Agent
    :param max_result_length: interger that used to mark the maximum length of result
    :return: string of page text content.
    """
    html = requests.get(url, headers=header)
    soup = BeautifulSoup(html.text, 'html.parser')

    result_list = soup.find_all('p')
    result = "\n".join(r.text for r in result_list)

    return result[:max_result_length]


def baidu_search(
    search: str,
    headers: dict = default_headers,
    search_site: str = None,
    search_max_results: int = 1,
    max_retry=3,
) -> list:
    """
    get web search results given by baidu, and get a list of web search results, each consists of title, url and text content.
    :param search: query
    :param headers: User-Agent
    :param search_site: specific the site you want to search on
    :param search_max_results: integer that used to mark the maxixum number of search results.
    :return: list of dictionaries, each consists of title, href, body.
    """
    url = search_url["baidu"]
    query = search if search_site is None else f"site:{search_site} " + search

    for _ in range(max_retry):
        response = requests.get(url=url + query, headers=headers)
        if response.status_code != 200:
            continue
        return _baidu_response_parse(response.text, search_site, search_max_results)

    return []

def bing_search(
    search: str,
    header = default_headers,
    search_max_results: int = 1,
    max_retry = 3,
) -> list:
    """
    get web search results given by bing, and get a list of web search results, each consists of title, url and text content.
    :param search: query
    :param headers: User-Agent
    :param search_max_results: integer that used to mark the maxixum number of search results.
    :return: list of dictionaries, each consists of title, href, body.
    """
    url = search_url["bing"]

    for _ in range(max_retry):
        response = requests.get(url=url + search, headers=header)
        if response.status_code != 200:
            continue
        return _bing_response_parse(html = response.text, search_max_results=search_max_results)
    return []

def wikipedia_search(
    search: str,
    search_max_results: int = 2,
    max_result_length: int = 500,
    max_retry: int = 3,
) -> list:
    """
    get web search results on wikipedia, and get a list of web search results, each consists of title, url and text content.
    :param search: query
    :param search_max_results: integer that used to mark the maxixum number of search results.
    :param max_result_length: interger that used to mark the maximum length of each results
    :return: list of dictionaries, each consists of title, href, body.
    """
    url = search_url["wikipedia"]
    header = {"User-Agent": "LangChainBot/0.0"}
    wiki_search = wikipediaapi.Wikipedia(header["User-Agent"], "en")
    query = search
    result_list = []

    for _ in range(max_retry):
        response = requests.get(url=url + query, headers=header)
        if response.status_code != 200:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        title_list = []
        search_result_list = soup.find_all(
            class_="mw-search-result mw-search-result-ns-0"
        )
        if search_result_list == []:
            page = wiki_search.page(query)
            if page.exists():
                return [
                    {
                        "title": query,
                        "href": None,
                        "body": page.summary[:max_result_length],
                    }
                ]
            else:
                return []
        else:
            for r in search_result_list[:search_max_results]:
                title_list.append(r.find("a")["title"])

            for title in title_list:
                page = wiki_search.page(title)
                if page.exists():
                    result_list.append(
                        {
                            "title": title,
                            "href": None,
                            "body": page.summary[:max_result_length],
                        }
                    )
            return result_list
    return []


def ddg_search(
    search: str,
    search_site: str = None,
    search_region: str = "wt-wt",
    search_max_results: int = 1,
) -> list:
    """
    get web search results given by duckduckgo, and get a list of web search results, each consists of title, url and text content.
    :param search: query
    :param search_site: specific the site you want to search on
    :param search_max_results: integer that used to mark the maxixum number of search results.
    :return: list of dictionaries, each consists of title, href, body.
    """
    query = search if search_site is None else f"site:{search_site} " + search
    result = []

    with DDGS() as ddgs:
        for r in ddgs.text(
            query,
            region=search_region,
            safesearch="off",
            timelimit="y",
            max_results=search_max_results,
        ):
            result.append(r)

    return result

def _bing_response_parse(
    html: str,
    header = default_headers,
    search_max_results: int = 1,
    max_result_length: int = 1000,
) -> list:
    """
    parse the html page given by bing, and get a list of web search results, each consists of title, url and text content.
    :param html: string of html page.
    :param header: User-Agent
    :param search_max_results: integer that used to mark the maxixum number of search results.
    :param max_result_length: interger that used to mark the maximum length of result
    :return: list of dictionaries, each consists of title, href, body.
    """
    soup = BeautifulSoup(html, "html.parser")
    search_num_results = 0  # the number of search results(successfully parsed)
    parsed_list = []

    result_list = soup.find_all(class_ = 'b_algo')

    for r in result_list:
        try:
            title = r.find('h2').text
            url = r.find('cite').text
            body = url_response_parse(url, header, max_result_length)
        except:
            url = ""
            title = ""
            body = ""

        if url != "" and title != "" and body != "":
            parsed_list.append({"title": title, "href": url, "body": body})
            search_num_results += 1
            if search_num_results >= search_max_results:
                break

    return parsed_list

def _baidu_get_real_url(v_url: str):
    """
    get the real address from virtual url by baidu
    :param v_url: 百度链接地址
    :return: 真实地址
    """
    r = requests.get(v_url, headers=default_headers, allow_redirects=False)  # 不允许重定向
    if r.status_code == 302:  # 如果返回302，就从响应头获取真实地址
        real_url = r.headers.get("Location")
    else:  # 否则从返回内容中用正则表达式提取出来真实地址
        real_url = re.findall("URL='(.*?)'", r.text)[0]
    # print('real_url is:', real_url)
    return real_url


def _baidu_response_parse(
    html: str,
    search_site: str = None,
    search_max_results: int = 1,
) -> list:
    """
    parse the html page given by baidu, and get a list of web search results, each consists of title, url and text content.
    :param html: string of html page.
    :param search_site: used to distinguish results between different search sites.
    :param search_max_results: integer that used to mark the maxixum number of search results.
    :return: list of dictionaries, each consists of title, href, body.
    """
    soup = BeautifulSoup(html, "html.parser")
    search_num_results = 0  # the number of search results(successfully parsed)
    parsed_list = []

    if search_site:
        result_list = soup.find_all(
            class_="result-op c-container new-pmd"
        ) + soup.find_all(class_="result-op c-container xpath-log new-pmd")
        for r in result_list:
            try:
                r_a = r.find("a")
                vurl = r_a["href"]
                url = _baidu_get_real_url(vurl)
                title = r_a.text
                body = r.find(class_="c-font-normal c-color-text").text
            except:
                url = ""
                title = ""
                body = ""

            if url != "" and title != "" and body != "":
                parsed_list.append({"title": title, "href": url, "body": body})
                search_num_results += 1
                if search_num_results >= search_max_results:
                    break
    else:
        result_list = soup.find_all(
            class_="result c-container new-pmd"
        ) + soup.find_all(class_="result c-container xpath-log new-pmd")
        for r in result_list:
            try:
                r_a = r.find("a")
                vurl = r_a["href"]
                url = _baidu_get_real_url(vurl)
                title = r_a.text
                body = r.find("span", class_="content-right_8Zs40").text
            except:
                url = ""
                title = ""
                body = ""

            if url != "" and title != "" and body != "":
                parsed_list.append({"title": title, "href": url, "body": body})
                search_num_results += 1
                if search_num_results >= search_max_results:
                    break

    return parsed_list


if __name__ == "__main__":
    import os

    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    print(
        web_search("what is apple", search_engine="bing", max_results=3)
    )
