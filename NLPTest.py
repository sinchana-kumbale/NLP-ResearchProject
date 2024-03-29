import xml.etree.ElementTree as ET
from datetime import time
from xml.dom import minidom

def toMinutes(start_time):
    mins = int(start_time[(start_time.index(':'))+1:])
    hrs = int(start_time[:(start_time.index(':'))])
    return (hrs*60 +mins)
tree = ET.parse(r'C:\Users\sinch\Desktop\Sinchana_Kumbale_DanceVideo\NLPData.xml')
root = tree.getroot()
usrconfig = []
print(len(root))
counter = 0
predatorList = ["00851429b21722a4d62f63a328c601ca","00aac10b39157377c79b7700b7b832bf","02800e11fdb1b43595303709f2b38f8c","03957f443c7790f9642db14bbc59df11","04bfa707d3313179ef48177d7270938e",
"04d42f7bb1eb41605dea74a8711f9fd0",
"0526eb9cfcee11c0036f3fa6d11158d5",
"053364a8ce3df76dadd5fe75fb056f72",
"0599cd3f7fc15849844468b0702ff593",
"0b6b05c740a1bf50ca7f9a461598a3b9",
"0cac6dfd241c5efe4ca07417575e582f",
"10e49cbbe257b19162a677113236cdc2",
"12140a644bf57166fe014116d1761ac0",
"1290ea419856093edf96b1263cb1ca1e",
"13ff9f88ad6e1242f30cb4c7d4a36ed7",
"18253a7ec92823c10255f80e57f88e74",
"18925613101337413b9752e383f46174",
"1d911cc805cce729269c8d357dc0f0fe",
"1e796ded59a56f729ecdf625e9007b7f",
"1eb17bd9642e93fa84969b71bf387a1b",
"1f3605cf14936ec8f6ef8d0527cff2c2",
"201a9319a4df100cb91e81644345f3b2",
"21d099c925ff6192643007e223ddc328",
"22b43adc75cd94c074892cee0b964eb8",
"23bffbccce4e395053495c68a5e15b68",
"249f9824cc41247e0eae1462151f2a8b",
"2918b2f2b8fe0d6f185ec6a2dd79c632",
"29db9baa96113d4d60886ab18027b6f8",
"2a1ac47332661b61d943d3a4e08dda5a",
"2a751a42619e8a40bf5115cb62135a69",
"2a77f862a19d8d47d21692543e90dee8",
"2d08262caa29b1bebeb24e9ad0c60ece",
"2e265f9b8ee76269872d56d5c6c0335b",
"303ddf8313a364e01e450d2b1af7e4f6",
"37da9b4030b08843cded697d709165ee",
"395b6188da0dcf2770643a7d79576aa5",
"39bd8b11509b5ddc5948b5b77f034ea1",
"3cf9477a8025ec6420d1e195e2b76324",
"3e48ec4a22b6f1f5b9a5bd991583294c",
"3e97c68b68f9aa0fb7d705a65c6a8443",
"404bc20efbd957d8a7c3986c3a53e202",
"422af212405c94b69bbe0c7ca9c8477c",
"47243a4a2c68f2f00899670d455a21fa",
"4851d3bd0d66d97ddb4437f0cb527a85",
"485483fff9616e27c7c8c88e53795d21",
"48aad4b6baaba42d5ad3b77880d3cead",
"4982b68761043e693da736df5852a7c5",
"4ab0a550ebb27f9a95e31ccbf0114850",
"4c33926fc6765490b09943d81ec86469",
"4cf8a879997585d68c7cdd49401b9e86",
"503343ecf680e0899a3863cc8bf2b9dc",
"54cad8b7d3538f5f60c1ffbc2058e945",
"57290165ecb488643f08da2c87e63550",
"5778ac8dcffe9bbacf3c0415fdb362d6",
"5ab85183f8a646d56077048b35679525",
"5d4180856d7162b6bd4549f412baba33",
"5f325f9eef9f7e6da2b3cd5717b416de",
"609f7b8e566e8d514eecf112d3d3bc95",
"62477e3c00adc3464999e6973e83fb52",
"6283fab0062e2eb309261acf647abd98",
"6465772a63c605b62764fd1c32cf6f1a",
"6d6c830b141a9716d2878f5126dc4516",
"6e9a4b8f988171cb6ccc88406fef2426",
"6f35a1f69fd4ae82056e4bc6a8a84575",
"6fa8da85e92704810cb756bfb3fd0441",
"70aca6a54d7d6b260273282143a685e0",
"72a17462620e221e26711493eda1fa1a",
"7dec33a7c4fc1d7295dea7cc5b966b5b",
"7fc0a10ac4f945ebf4004b258179ad1c",
"80706012f8f9f1175c8e37c306394727",
"81383171ca9da4d245dea449027c3d09",
"821f0b4dcb8c29c1852f400f838ce7ba",
"850f47f5c88b660f1b096785ddc197a1",
"85346c8846fc694078665377fec333b0",
"872ba632599e4f3ff3a4921da8ae3da8",
"89319407d854bb82113349ecf7ce3682",
"8cd850ea4215ee7c4b94b6bcc0bae593",
"9024719626a23e08ec06430a8a36cb72",
"902cddc2765a0700c9d1d543c9a4d52f",
"90dfa20487af303d92fe68ec15eb1e3e",
"926981922f4a48d98ebc92a9375e5d45",
"94b4cd6904fd60f7ac5d3bdbe98aeb62",
"9536bdac0e1351f316701b9310236df0",
"965a47c45807e6a0b92c9ef340eb5f62",
"970f6cbfa8b79242464120ec1bc7d074",
"9775a796b1ac00a2a756f530bf554582",
"9b227243286fca8a81bd41ce8551d662",
"9dc04852863541e939e7c66cf5469240",
"9f1180d3743c2e880a36aaefbb0a80ef",
"a03edc2f70bbebc73ef3ba3f06968360",
"a219f094e5c784b090f60097c84a6ed2",
"aa98bc7761802289d80ec7096028251d",
"ab9d9cb1746f4aa471cb7c648d6cbefa",
"abf6d5f2a3a6a9ad3322c0e704cb8107",
"aebe7a87ad50fbad92d9931c49cae848",
"af7a0adebf50023598764bdaf2080be4",
"afe8ba8af6d0677a79ce0284cb9cc63a",
"b4a8480c88776ca09e50c327a56f5e16",
"b5486deb3c96fa5fe9d40aa30fdf61b5",
"b679fca2e3690b4d3c60815edf4e3ca5",
"ba22656bb7e05cf53bedf6e279b76871",
"ba5cfb3cabb6d0f115824c728a6d36bb",
"bb0ecb9641f6d19a264e835f465d8a9e",
"bc67b0731515b5e375e6c9aee82bad84",
"bdaa4b675589e3b8f93f1655fed3b314",
"bfa317c5529e65c5897c4cd4871e35d0",
"c1b9bdab71d1c56640da37d5e0f5261d",
"c1f43e1c992120a1c460b8ef7d0b4e9b",
"c420f4c2451ed50149332783dd90db59",
"c62283536cf6261e5ffbcb323c8a2571",
"c9f247cf065f29af8f568d313284d9e1",
"ca2ac2fc60098c8f3346aed97ccacbec",
"ca5496597b7151b9c2414c16c4ef5422",
"cc5e22dc487f37deebdcf482a2c22fd8",
"ce08e214c3a26b53ade62af0a8c453cc",
"cfd5fabe2c8381ad5006a5cfc471a050",
"d033ef02538095a94f42313f519bed88",
"d090448efc59f5f5b3b5c2875d070a3a",
"d18fb2dc834414a71aace67bee91c432",
"d43e87eedeb586fe9431568921e2fdf7",
"d50f114dde2edb12b72ecea83ebf63ce",
"d6af75e4a889d4c27383ab51b6fd07af",
"d79fcf39ef0828bb234b60b90a1fd725",
"db12fc3e76fd54f68185a16423a7325b",
"df00009d297a68f5a0f1967d9e509bb1",
"dfcb21b491de12ccfa7703216e646d3d",
"e04c2ebb13344e6da4e4a280d36e8425",
"e1b76a292497409f9deadbe07d1f7e6d",
"e4ba84a61fda7a2fe55ea050aef7a26c",
"e4c7c376bbd07aeb4a59684a2b94a664",
"e67a150cbffcef5310a8c38f73281526",
"e73f6d03ba8e592e9e59dd69b14cf6b9",
"e788b3a47e4d812ed26036a93272c134",
"ed4ece301c57d14e22b2212e5a7f25ca",
"f069dbec9ab3e090972d432db279e3eb",
"f1c7705309a41293e04791e994d4b7b3",
"f506fdbb8632c8b7b5d7c3203f702699",
"f538a2a92145534b4b61893399479fcf",
"f5e20f1c82066b5d6801d889d146c4f9",
"f91e272c9481c92c08505fffbe726053",
"fac3a2081264f1dbb943eaf7165d8fc3",
"fb169d16c42ca83583dd05b3f4d7fc98"]

for i in root:
    participants = []
    for key in i.findall('message'):
        p = key.find('author').text
        participants.append(p)
    if (any(check in participants for check in predatorList)):
        counter += 1

        
    start_time = i[0][1].text
    minutes = int(start_time[(start_time.index(':'))+1:]) + 45
    if (minutes>60):
        end_time = str(int(start_time[:(start_time.index(':'))])+1).zfill(2)+":"+str(minutes-60).zfill(2)
    else:
        end_time = str(int(start_time[:(start_time.index(':'))])).zfill(2)+":"+str(minutes).zfill(2)
    start_seconds = toMinutes(start_time)
    end_seconds = toMinutes(end_time)
    if (end_seconds < toMinutes(i[-1][1].text)):
        k = 0
        while ((toMinutes(i[k][1].text))<= end_seconds):
            
            lst.append(i[k])
            k += 1
        for p in i:
            if p not in lst:
                i.remove(p)
        lst = i
    else:
        lst = i
        
    usrconfig.append(lst)
print(participants)
print(counter)

with open('output.xml','wb') as f:
    for i in usrconfig:
            f.write(ET.tostring(i))


        
    
