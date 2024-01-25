import re, requests, time  # 导入所需要的库

headers = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Mobile Safari/537.36",
    "Cookie": "BAIDUID=229A18B7534A5CEA671381D45FCDC530:FG=1; BIDUPSID=229A18B7534A5CEA671381D45FCDC530; PSTM=1592693385; BDRCVFR[dG2JNJb_ajR]=mk3SLVN4HKm; userFrom=null; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm; H_WISE_SIDS=149389_148867_148211_149537_146732_138426_150175_147527_145599_148186_147715_149253_150045_149280_145607_148660_146055_110085; delPer=0; BDORZ=AE84CDB3A529C0F8A2B9DCDD1D18B695; ysm=10315; IMG_WH=626_611; __bsi=8556698095607456048_00_14_R_R_17_0303_c02f_Y",
}

detail_urls = []  # 存储图片地址

for i in range(1, 400, 20):  # 20页一张
    url = 'http://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=&st=-1&fm=result&fr=&sf=1&fmq=1592804203005_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1592804203008%5E00_1328X727&sid=&word=瓢虫&pn={}'.format(
        i)  # 请求的地址
    response = requests.get(url, headers, timeout=(3, 7))  # 设置请求超时时间3-7秒
    content = response.content.decode('utf-8')  # 使用utf-8进行解码
    detail_url = re.findall('"objURL":"(.*?)"', content, re.DOTALL)  # re.DOTALL忽略格式#匹配objURL的内容,大部分为objURL或URL
    detail_urls.append(detail_url)  # 将获取到的图片地址保存在之前定义的列表中
    response = requests.get(url, headers=headers)  # 请求网站
    content = response.content
b = 0  # 图片第几张
for page in detail_urls:
    for url in page:
        try:
            print('获取到{}张图片'.format(i))
            response = requests.get(url, headers=headers)
            content = response.content
            if url[-3:] == 'jpg':
                with open('./train_insects/瓢虫/{}.jpg'.format(b), 'wb') as f:
                    f.write(content)
            else:
                continue

        except:
            print('超时')
        b += 1

