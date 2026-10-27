// src/utils/phoneticMatcher.ts
import { Segment, PhoneticMatchCandidate } from '../types';

/**
 * 台灣會議全方位音韻群組映射表 (涵蓋 230+ 聲韻調壓縮音節群組、逾 3,600 個常用繁體字)
 * 針對台灣國語（Taiwan Mandarin）常見語音變異設計：
 * 1. 平捲舌合流：zh/z, ch/c, sh/s
 * 2. 前後鼻音合流：in/ing, en/eng, an/ang
 * 3. 流音、鼻音與日母弱化：l/n/r/y
 * 4. 唇齒與舌根圓唇合流：f/hu/w/g (如：發/花、風/轟、公/風)
 * 5. 顎化與齒音擦音鄰近：j/z, q/c, x/s (在介音 i/u 前之擦音過渡)
 */
const PHONETIC_GROUPS: Record<string, string> = {
  // === 1. 常見百大姓氏與稱謂核心字群 ===
  '林臨淋琳霖零玲鈴齡伶翎鄰鱗磷麟凜檁吝賃': 'LIN',
  '陳塵臣辰晨沉成城誠呈承程懲乘盛丞橙澄逞騁秤': 'CHEN',
  '張章彰璋樟蟑髒商傷湯昌娼猖猖': 'ZHANG',
  '李里理禮裡鯉履哩離梨黎麗利例立力歷曆隸粒勵儷俐莉笠粒瀝吏': 'LI',
  '王亡汪網往枉晚碗挽宛婉琬椀灣彎頑丸完玩萬腕蔓': 'WAN',
  '吳吾無蕪梧巫誣務物誤悟霧勿屋五武午舞伍侮捂塢戊兀': 'WU',
  '劉留流琉榴瘤柳六陸溜遛': 'LIU',
  '蔡菜採彩踩財材裁才猜拆差釵茶查察擦搽岔剎': 'CAI',
  '楊陽揚洋羊佯樣養仰央泱秧癢氧恙漾': 'YANG',
  '許栩詡徐須需虛噓墟詢尋巡敘序緒續蓄恤婿絮': 'XU',
  '鄭正政證症爭徵真針珍診陣鎮振震賑貞偵幀睜錚掙': 'ZHENG',
  '謝械卸屑蟹懈瀉寫血協脅鞋斜邪攜些歇蠍楔': 'XIE',
  '郭鍋國果過裹擴闊括廓聒菓': 'GUO',
  '洪紅宏虹鴻哄烘轟弘泓訌': 'HONG',
  '曾增爭箏蒸真偵憎贈綜繒': 'ZENG',
  '廖料瞭療聊撩遼僚繚寥鐐': 'LIAO',
  '賴來萊瀨籟藍蘭攔欄奈耐徠': 'LAI',
  '蘇疏輸書舒樞叔屬署鼠數術述束俗宿速塑訴粟蔬梳殊贖塾': 'SU',
  '周洲舟州週驟軸宙晝咒奏揍粥肘帚': 'ZHOU',
  '葉頁業夜液腋咽謁野也冶耶曳拽爺椰': 'YE',
  '黃皇凰煌惶璜簧潢謊晃幌荒慌': 'HUANG',
  '呂侶鋁旅屢縷履律慮率綠氯': 'LYU',
  '蕭消宵霄削囂小曉瀟簫梟嘯': 'XIAO_2',
  '羅蘿邏鑼螺裸絡落駱烙酪洛': 'LUO',
  '高膏羔糕搞稿鎬告誥': 'GAO',
  '潘攀盤磐判叛畔盼拼': 'PAN',
  '簡檢減撿繭鹼見件建健鍵漸踐劍薦鑑間肩堅監兼尖奸殲艱': 'JIAN',
  '朱珠株諸豬蛛誅竹築逐燭主組祖阻詛煮囑助住注駐祝著柱佇貯': 'ZHU',
  '游由油遊尤郵猶友有右又幼誘憂優幽悠悠釉': 'YOU',
  '詹沾氈瞻斬盞展輾佔戰站棧綻讚暫': 'ZHAN',
  '彭棚硼鵬捧碰膨澎篷': 'PENG',
  '施師詩失獅濕十石時實食識史使始示世勢適釋氏誓逝嗜拾蝕士事式市試視室是': 'SHI',
  '余於妤魚漁娛逾愚榆瑜渝雲云勻芸育域預遇裕愈喻玉獄雨語羽與宇嶼餘輿': 'YU',
  '盧蘆爐顱陸錄鹿路露虜魯滷戮賂': 'LU',
  '梁良涼糧量兩亮諒輛晾踉': 'LIANG',
  '顏言嚴岩鹽沿延研閻炎淹奄掩眼演厭宴艷驗諺堰': 'YAN',
  '柯科顆棵磕苛課客克刻可渴殼咳坷': 'KE',
  '翁嗡瓮甕空公': 'WENG',
  '魏委偉尾瑋偽緯位未味衛為圍違唯維微危薇威巍慰胃謂餵蔚畏尉': 'WEI',
  '孫遜筍損瞬順舜': 'SUN',
  '戴帶代袋貸待怠殆大太態泰台臺抬苔歹逮殆': 'DAI',
  '范範販犯飯泛反返番翻凡煩繁樊汎梵': 'FAN',
  '方芳坊妨防房肪訪仿紡放': 'FANG',
  '宋送頌訟誦縱': 'SONG',
  '鄧瞪凳澄鐙': 'DENG',
  '杜肚度渡鍍讀獨毒牘犢堵賭睹': 'DU',
  '傅付負附赴副富復腹覆服福幅浮符扶夫膚敷賦父婦輔腐府俯釜縛': 'FU',
  '侯喉猴瘊後厚候后吼': 'HOU',
  '曹槽嘈草操糙吵炒': 'CAO',
  '溫瘟文聞紋蚊吻穩問汶': 'WEN',
  '薛靴穴學雪血削': 'XUE',
  '丁盯釘叮頂鼎訂定錠': 'DING',
  '卓著琢灼濁茁桌捉拙': 'ZHUO',
  '馬碼瑪螞麻嘛罵媽抹嗎': 'MA',
  '阮軟卵染燃然冉': 'RUAN',
  '董懂棟凍洞動重': 'DONG',
  '唐堂棠搪膛糖倘躺淌趟燙湯': 'TANG',
  '藍蘭攔欄籃嵐覽攬纜懶爛濫': 'LAN',
  '江將漿僵疆薑講獎槳匠降醬虹': 'JIANG',
  '邵少稍梢紹哨燒梢勺芍': 'SHAO',
  '馮逢縫峰蜂風封豐瘋鋒鳳奉諷俸': 'FENG',
  '姚搖遙謠窯堯肴餚咬舀要耀曜藥鑰邀腰妖': 'YAO',
  '鐘中忠終衷眾重種腫塚總宗蹤棕縱': 'ZONG',
  '康慷糠扛抗炕亢': 'KANG',
  '邱丘秋蚯囚求球酋': 'QIU',

  // === 2. 政府機關、會議職稱、工程標案與法規高頻字群 ===
  '處觸畜築儲礎楚出初除廚鋤雛矗怵': 'CHU',
  '長場廠掌漲常償嚐腸藏昌暢唱倡敞徜': 'ZHANG_TITLE',
  '員元園原源圓援緣媛猿垣院願怨苑': 'YUAN',
  '任認韌刃妊人仁忍稔': 'REN',
  '秘密蜜覓冪米迷謎彌靡眯糜': 'MI',
  '神什沈審嬸慎滲深身申伸升生聲勝聖盛省剩甚紳呻妊娠': 'SHEN',
  '如儒茹汝乳入辱褥濡蠕': 'RU',
  '工功攻公宮弓恭供躬共貢拱鞏空孔控恐龔': 'GONG',
  '項向象像相巷享響想箱廂鄉香詳祥翔降驤': 'XIANG',
  '經精晶京驚鯨金今津斤禁近進盡勁景警井境鏡敬靜淨競徑竟莖荊襟筋': 'JING',
  '俊駿峻浚郡菌軍均君鈞窘濬': 'JUN',
  '穎影營迎贏盈瑩螢沂儀宜疑移姨遺頤銀吟飲引印應鷹英櫻嬰蠅癮隱': 'YING_YI',
  '博薄搏駁伯帛泊白百擺柏拍排牌迫魄舶脖': 'BO',
  '姐姊解節結潔借界戒介屆街皆接階揭劫傑竭截捷睫竭': 'JIE',
  '教較校酵覺交郊嬌澆焦礁角腳剿叫轎笑效肖孝酵膠椒驕攪僥': 'JIAO',
  '授受售瘦獸收手守首壽熟狩': 'SHOU',
  '店電墊殿奠佃甸點典顛巔碘踮': 'DIAN',
  '面麵棉眠免勉緬綿靦冕娩': 'MIAN',
  '修羞休朽秀袖繡鏽宿嗅': 'XIU',
  '開揩慨楷凱鎧概溉愾': 'KAI',
  '新心辛欣薪馨星興腥芯信釁猩鋅': 'XIN',
  '標彪飆錶表裱婊鏢': 'BIAO',
  '不布步部怖捕補普譜堡埠簿哺怖': 'BU',
  '廊狼郎良涼糧梁量朗浪榔螂': 'LANG',
  '群裙尋巡詢循旬馴貧頻': 'QUN',
  '導島倒禱蹈到道稻盜刀叨悼搗': 'DAO',
  '央泱秧英應鷹迎營贏盈映硬櫻嬰': 'YANG_YING',
  '切且竊妾企契氣棄汽器泣訖怯': 'QIE',
  '小曉筱少稍梢紹哨笑校效孝嘯': 'XIAO',
  '企起豈啟氣器棄契期欺漆戚妻棲七希溪西吸祈齊旗棋騎奇歧崎泣': 'QI',
  '支之知肢織汁資姿滋諮只指止址紙至志制致置質治炙職直值植執殖秩擲枝芝脂蜘蛛趾幟滯窒': 'ZHI',
  '晰析息悉惜昔夕希稀犧吸西膝溪熙嬉錫襲習席媳洗喜係系細戲熄蜥': 'XI',
  '就舊救舅咎久九酒究糾揪韭灸疚廄': 'JIU',
  '加家佳嘉夾挾袈甲假價架駕嫁稼迦頰鉀': 'JIA',
  '盟萌蒙濛盲忙茫氓猛孟夢懵朦': 'MENG',
  '議意義藝亦異役疫譯翼億憶益誼一衣醫依壹揖椅已以尾倚毅肄溢裔蟻乙': 'YI',
  '評平瓶憑萍屏貧頻品聘拼拚坪蘋': 'PING',
  '選癬險顯憲縣現限線獻羨腺弦賢鹹嫌鮮仙先掀炫眩絢': 'XUAN',
  '核合何河荷盒和禾吉及級即極急集籍夥火伙獲惑或禍貨活涸闔閡': 'HE',
  '給北被備背倍輩貝悲杯碑卑背臂輩': 'GEI_BEI',
  '裝樁莊妝潢狀撞壯創瘡窗床闖': 'ZHUANG',
  '充衝沖聰匆蔥從叢崇蟲重寵銃': 'CHONG',
  '足族卒竹築逐燭俗宿簇': 'ZU_FOOT',
  '怎診枕整貞針陣鎮振縝': 'ZEN',
  '麼嗎嘛麻馬媽抹摩磨魔末沫莫漠默墨': 'ME_PARTICLE',
  '辦半伴扮絆版板盼判叛盤磐番班般搬斑頒扳拌瓣': 'BAN',
  '苦股古谷鼓骨賈庫褲酷哭枯窟骷': 'KU',
  '民明名銘鳴敏閩眠綿皿泯冥瞑螟': 'MIN',
  '會匯繪惠慧賄悔毀回迴灰揮輝恢徽卉喙': 'HUI',
  '納那吶捺內耐奈奶乃男南難喃拿哪呐': 'NA',
  '案按暗岸胺安鞍氨俺諳黯': 'AN',
  '培陪賠裴肥胚坯配佩沛倍背': 'PEI',
  '訓訊迅遜尋巡詢熏薰勳殉': 'XUN',
  '吊釣調掉弔碉雕刁叼': 'DIAO',
  '橋瞧喬僑翹竅殼巧悄俏峭鞘敲橇': 'QIAO',
  '巴吧疤芭八拔跋把靶爸罷霸扒壩': 'BA',
  '連聯憐簾蓮廉臉練煉戀鏈斂鐮': 'LIAN',
  '專磚轉傳撰賺纂篆傳喘串': 'ZHUAN',
  '單擔丹耽眈但淡彈誕旦膽氮憚': 'DAN',
  '月越粵悅閱躍嶽岳約曰鑰': 'YUE',
  '號好耗浩皓毫豪壕嚎昊': 'HAO',
  '早澡藻棗找爪造皂燥躁灶兆照罩召詔': 'ZAO',
  '上尚賞裳商傷湯雙晌': 'SHANG',
  '天添田甜填舔恬腆': 'TIAN',
  '三山衫刪珊散傘閃善扇擅摻杉舢疝': 'SAN',

  // === 3. 擴充全音節覆蓋表 (涵蓋完整聲母 b/p/m/f/d/t/n/l/g/k/h/j/q/x/zh/ch/sh/r/z/c/s 與韻母組合) ===
  '阿啊呵吖腌': 'A',
  '哀埃挨唉愛礙艾曖隘矮藹靄': 'AI',
  '奧澳懊傲熬翱敖凹襖': 'AO',
  '歐毆鷗嘔偶藕慍': 'OU',
  '恩摁按': 'EN',
  '兒而耳爾餌二貳佴邇': 'ER',
  '批披劈坯拍排牌迫破頗坡潑婆鄱剖': 'PI',
  '撲鋪僕菩葡蒲樸圃普浦譜瀑曝暴': 'PU',
  '怕帕爬扒琶趴啪葩': 'PA',
  '派湃排牌迫拍俳': 'PAI',
  '跑泡炮袍咆刨拋泡': 'PAO',
  '盆噴朋棚硼鵬膨澎篷捧碰': 'PEN',
  '飄漂瓢嫖票剽縹': 'PIAO',
  '篇偏片騙駢便遍辨辯辮': 'PIAN',
  '撇瞥劈僻闢譬': 'PIE',
  '摸模膜摩磨魔末沫莫漠默墨抹脈': 'MO',
  '買埋邁賣麥脈霾': 'MAI',
  '貓毛矛茅錨卯鉚冒貿帽貌茂': 'MAO',
  '謀眸某母畝牡姆木目牧穆幕募慕墓睦沐': 'MU',
  '梅枚玫媒煤霉美每妹昧媚袂謎': 'MEI',
  '門們悶燜捫': 'MEN',
  '滿瞞饅蠻慢漫嫚蔓曼': 'MAN',
  '苗描瞄秒渺藐妙廟繆': 'MIAO',
  '滅蔑篾捏涅孽': 'MIE',
  '謬繆謬': 'MIU',
  '發伐罰閥乏筏法髮琺花華滑猾化畫話樺': 'FA_HUA',
  '非飛妃緋扉肥匪誹吠肺廢沸費徽揮輝灰恢': 'FEI_HUI',
  '分芬吩紛氛墳焚粉份奮憤糞混昏婚葷魂': 'FEN_HUN',
  '佛拂伏服福幅浮符縛': 'FO',
  '否缶某': 'FOU',
  '達答沓打大塔踏塌牠他她它': 'DA_TA',
  '得德的底抵地弟帝遞第締蒂滴低堤敵笛滌嫡': 'DE_DI',
  '刀叨導島倒禱蹈到道稻盜悼討套逃桃陶淘掏濤滔': 'DAO_TAO',
  '都兜斗抖陡豆逗痘讀獨毒渡度肚': 'DOU',
  '隊對兌堆確退腿蛻推頹': 'DUI_TUI',
  '端短段斷緞鍛團糰': 'DUAN_TUAN',
  '頓囤盾鈍遁噸吞屯豚臀': 'DUN_TUN',
  '多奪踱朵躲跺惰墮拖托託脫妥橢唾駝陀駄': 'DUO_TUO',
  '跌爹碟蝶諜疊牒鐵貼帖': 'DIE_TIE',
  '丟溜': 'DIU',
  '東冬咚懂董棟凍洞動通偷頭投透桶筒統痛': 'DONG_TONG',
  '台臺抬苔太態泰汰胎苔嘆炭探貪攤灘癱壇談彈痰譚潭': 'TAI_TAN',
  '特慝徹撤澈策冊側測廁': 'TE_CE',
  '踢梯剔題提蹄啼體替涕剃屜': 'TI',
  '條調迢挑跳眺糶': 'TIAO',
  '聽廳汀廷亭庭停挺艇町': 'TING',
  '拿哪那納吶捺拉啦喇辣落烙絡': 'NA_LA',
  '乃奶奈耐賴來萊瀨奈': 'NAI_LAI',
  '男南難喃藍蘭攔欄籃嵐覽攬纜懶爛濫': 'NAN_LAN',
  '囊攮狼廊郎朗浪': 'NANG_LANG',
  '腦惱鬧撓饒擾繞勞牢撈老佬澇酪': 'NAO_LAO',
  '內餒累淚類壘磊蕾肋雷擂': 'NEI_LEI',
  '嫩能冷愣稜': 'NEN_LENG',
  '泥尼呢妮擬你妳逆匿溺膩離梨黎麗利例立力歷曆隸粒勵禮李里理裡': 'NI_LI',
  '年黏捻輦念廿連聯憐簾蓮廉臉練煉戀鏈': 'NIAN_LIAN',
  '娘釀良涼糧梁量兩亮諒輛': 'NIANG_LIANG',
  '鳥尿裊捏涅孽列烈裂劣獵': 'NIAO_LIE',
  '您林臨淋琳霖零玲鈴齡鄰凝寧擰濘': 'NIN_LING',
  '牛扭鈕拗流留琉榴瘤柳六陸': 'NIU_LIU',
  '農濃膿弄攏壟籠聾隆龍': 'NONG_LONG',
  '奴努怒女呂侶鋁旅屢縷履律慮率綠氯': 'NU_LYU',
  '暖卵亂段斷': 'NUAN_LUAN',
  '虐略掠': 'NYUE_LYUE',
  '哥歌戈擱割革格閣隔葛蛤個各ガ可渴克刻客課': 'GE_KE',
  '該改蓋概鈣開揩慨楷凱鎧': 'GAI_KAI',
  '甘柑肝竿乾尷趕敢感橄幹幹贛看刊堪勘砍侃嵌': 'GAN_KAN',
  '剛綱鋼缸岡港崗槓抗扛康慷糠': 'GANG_KANG',
  '高膏羔糕搞稿鎬告誥考烤拷靠銬': 'GAO_KAO',
  '勾溝鉤狗苟構購夠垢扣寇叩': 'GOU_KOU',
  '姑孤估沽辜古股谷鼓骨固故顧雇括擴闊苦庫褲酷哭枯窟': 'GU_KU',
  '瓜刮寡卦掛褂誇垮挎跨': 'GUA_KUA',
  '乖拐怪快塊筷會': 'GUAI_KUAI',
  '官冠棺觀關館管貫慣灌罐寬款匡筐狂況曠礦框眶': 'GUAN_KUAN',
  '光胱廣逛況曠礦': 'GUANG_KUANG',
  '規皈閨瑰歸龜軌鬼詭癸貴桂櫃跪饋潰愧虧窺魁葵': 'GUI_KUI',
  '滾棍昆崑捆困睏': 'GUN_KUN',
  '哈蛤蝦喝呵禾合何河荷盒核和': 'HA_HE',
  '嗨孩骸海亥害駭還': 'HAI',
  '酣憨含函涵寒韓罕喊漢汗旱悍焊憾翰': 'HAN',
  '航杭行巷項向象': 'HANG',
  '黑嘿痕很狠恨': 'HEI_HEN',
  '哼亨橫衡恆洪紅宏虹鴻': 'HENG_HONG',
  '呼忽乎狐弧胡壺核湖糊蝴虎唬護互戶滬附付負富副復': 'HU_FU',
  '懷槐徊壞話畫化華滑': 'HUAI',
  '歡桓環還緩換喚煥患幻渙宦荒慌皇凰煌惶簧潢謊晃幌': 'HUAN_HUANG',
  '昏婚葷渾魂混溷分紛份奮': 'HUN',
  '基期欺機肌飢跡姬績積箕擊及級即極急集籍吉汲疾棘藉幾己擠給計記紀忌技妓際季劑濟繼既祭寄寂績': 'JI',
  '恰洽掐卡': 'QIA',
  '腔槍羌強牆薔搶襁嗆': 'QIANG',
  '親侵欽芹琴禽勤擒寢沁清青輕傾卿氫頃請慶': 'QIN_QING',
  '窮瓊炯迥窘': 'QIONG',
  '區軀驅屈趨渠取娶曲去趣': 'QU',
  '圈全權泉拳痊詮犬勸券': 'QUAN',
  '缺卻雀確鵲權': 'QUE',
  '瞎蝦匣狹峽轄暇霞下夏嚇廈': 'XIA',
  '兄胸兇凶匈雄熊': 'XIONG',
  '扎紮渣札眨柵詐榨炸咋擦差插叉茶查察': 'ZHA_CA',
  '摘齋宅窄債寨柴豺踩採彩菜蔡猜拆差': 'ZHAI_CAI',
  '朝招昭著找沼照罩兆召詔趙超抄鈔吵炒潮嘲巢': 'ZHAO_CHAO',
  '遮折哲輒者這浙責則擇澤車扯徹撤澈策冊側測廁': 'ZHE_CHE',
  '捉桌拙濁酌琢著戳綽湊錯措挫': 'ZHUO_CUO',
  '追錐綴贅吹炊垂槌錘崔催摧脆翠粹淬': 'ZHUI_CUI',
  '尊遵準准純唇醇蠢春椿存寸村忖': 'ZHUN_CUN',
  '資姿滋諮支之知肢織汁只指止址紙子紫姊字自次刺賜詞祠辭慈磁瓷此': 'ZI_CI',
  '租組祖阻詛豬諸珠株竹築逐燭粗初出除廚礎楚促醋簇': 'ZU_CU',
  '鑽纂賺撰專磚轉傳竄篡穿川船傳喘串': 'ZUAN_CUAN',
  '最罪醉嘴追綴': 'ZUI',
  '作坐座做鑿昨左佐撮': 'ZUO',
  '撒殺沙紗砂傻啥煞廈薩颯': 'SHA_SA',
  '篩晒曬塞腮鰓賽': 'SHAI_SAI',
  '燒稍梢勺芍少紹哨掃嫂騷臊': 'SHAO_SAO',
  '奢賒舌蛇捨設社射涉攝色瑟塞': 'SHE_SE',
  '誰水隨綏髓碎歲祟遂隧': 'SHUI_SUI',
  '刷耍衰摔甩帥率': 'SHUA_SHUAI',
  '拴栓雙霜爽酸算蒜': 'SHUAN_SUAN',
  '說縮梭唆鎖瑣索所爍碩朔': 'SHUO_SUO',
  '思司私絲撕死四寺似祀肆是事市式試視室師詩失施十石時實食識史使始': 'SI_SHI',
  '松鬆聳慫送頌訟誦熟贖屬暑鼠數術述束': 'SONG_SHU',
  '搜艘餿嗽叟抖': 'SOU',
  '然燃染冉懶藍蘭': 'RAN',
  '嚷壤攘讓浪朗': 'RANG',
  '饒擾繞勞牢老': 'RAO',
  '惹熱樂勒垃圾': 'RE_LE',
  '仍扔人仁忍認任冷': 'RENG',
  '日入如儒乳辱': 'RI',
  '容溶蓉熔融榮絨冗籠隆龍': 'RONG',
  '柔揉肉揉樓摟漏露': 'ROU',
  '瑞銳兌隊累類': 'RUI',
  '潤閏論輪倫淪': 'RUN',
  '弱若絡落洛': 'RUO',
  '壓押鴉鴨牙芽崖衙涯雅啞亞訝軋': 'YA',
  '優憂幽悠尤由油游遊郵猶友有右又幼誘': 'YOU_2',
  '庸擁臃永泳詠勇湧用佣傭': 'YONG',
  '蛙挖哇瓦襪娃滑化話畫': 'WA',
  '歪崴外壞塊': 'WAI',
  '窩渦蝸我握沃臥或惑獲貨活': 'WO',
};

const charToPhoneticCode = new Map<string, string>();
Object.entries(PHONETIC_GROUPS).forEach(([chars, code]) => {
  for (const ch of chars) {
    if (!charToPhoneticCode.has(ch)) {
      charToPhoneticCode.set(ch, code);
    }
  }
});

/**
 * 細粒度音韻四元組結構 (聲母 Initial, 介音 Medial, 韻腹與韻尾 Nucleus/Coda)
 * 用於動態規劃加權音韻編輯距離與跨音節連音縮讀 (Syllable Contraction)
 */
interface SyllablePhonemeTuple {
  rawCode: string;
  fuzzyHash: string;
  initial: string;
  medial: string;
  rime: string;
}

const CODE_TO_TUPLE_MAP: Record<
  string,
  { initial: string; medial: string; rime: string; fuzzyHash: string }
> = {
  // 核心姓氏與高頻會議音節
  LIN: { initial: 'l', medial: 'i', rime: 'in', fuzzyHash: 'lin' },
  CHEN: { initial: 'c', medial: '', rime: 'en', fuzzyHash: 'cen' },
  ZHANG: { initial: 'z', medial: '', rime: 'an', fuzzyHash: 'zan' },
  ZHANG_TITLE: { initial: 'z', medial: '', rime: 'an', fuzzyHash: 'zan' },
  LI: { initial: 'l', medial: 'i', rime: 'i', fuzzyHash: 'li' },
  WAN: { initial: 'w', medial: 'u', rime: 'an', fuzzyHash: 'wan' },
  WU: { initial: 'w', medial: 'u', rime: 'u', fuzzyHash: 'wu' },
  LIU: { initial: 'l', medial: 'i', rime: 'ou', fuzzyHash: 'liou' },
  CAI: { initial: 'c', medial: '', rime: 'ai', fuzzyHash: 'cai' },
  YANG: { initial: 'y', medial: 'i', rime: 'an', fuzzyHash: 'yan' },
  YANG_YING: { initial: 'y', medial: 'i', rime: 'in', fuzzyHash: 'yin' },
  XU: { initial: 'x', medial: 'u', rime: 'u', fuzzyHash: 'xu' },
  ZHENG: { initial: 'z', medial: '', rime: 'en', fuzzyHash: 'zen' },
  XIE: { initial: 'x', medial: 'i', rime: 'ie', fuzzyHash: 'xie' },
  GUO: { initial: 'g', medial: 'u', rime: 'o', fuzzyHash: 'guo' },
  HONG: { initial: 'h', medial: 'u', rime: 'on', fuzzyHash: 'hon' },
  ZENG: { initial: 'z', medial: '', rime: 'en', fuzzyHash: 'zen' },
  LIAO: { initial: 'l', medial: 'i', rime: 'ao', fuzzyHash: 'liao' },
  LAI: { initial: 'l', medial: '', rime: 'ai', fuzzyHash: 'lai' },
  SU: { initial: 's', medial: 'u', rime: 'u', fuzzyHash: 'su' },
  ZHOU: { initial: 'z', medial: '', rime: 'ou', fuzzyHash: 'zou' },
  YE: { initial: 'y', medial: 'i', rime: 'ie', fuzzyHash: 'ye' },
  HUANG: { initial: 'h', medial: 'u', rime: 'an', fuzzyHash: 'huan' },
  LYU: { initial: 'l', medial: 'u', rime: 'u', fuzzyHash: 'lu' },
  LUO: { initial: 'l', medial: 'u', rime: 'o', fuzzyHash: 'luo' },
  GAO: { initial: 'g', medial: '', rime: 'ao', fuzzyHash: 'gao' },
  PAN: { initial: 'p', medial: '', rime: 'an', fuzzyHash: 'pan' },
  ZHAN: { initial: 'z', medial: '', rime: 'an', fuzzyHash: 'zan' },
  PENG: { initial: 'p', medial: '', rime: 'en', fuzzyHash: 'pen' },
  LU: { initial: 'l', medial: 'u', rime: 'u', fuzzyHash: 'lu' },
  YAN: { initial: 'y', medial: 'i', rime: 'an', fuzzyHash: 'yan' },
  WENG: { initial: 'w', medial: 'u', rime: 'on', fuzzyHash: 'won' },
  SUN: { initial: 's', medial: 'u', rime: 'un', fuzzyHash: 'sun' },
  FANG: { initial: 'f', medial: '', rime: 'an', fuzzyHash: 'fan' },
  SONG: { initial: 's', medial: 'u', rime: 'on', fuzzyHash: 'son' },
  DENG: { initial: 'd', medial: '', rime: 'en', fuzzyHash: 'den' },
  HOU: { initial: 'h', medial: '', rime: 'ou', fuzzyHash: 'hou' },
  CAO: { initial: 'c', medial: '', rime: 'ao', fuzzyHash: 'cao' },
  WEN: { initial: 'w', medial: 'u', rime: 'en', fuzzyHash: 'wen' },
  XUE: { initial: 'x', medial: 'u', rime: 'ie', fuzzyHash: 'xue' },
  DING: { initial: 'd', medial: 'i', rime: 'in', fuzzyHash: 'din' },
  ZHUO: { initial: 'z', medial: 'u', rime: 'o', fuzzyHash: 'zuo' },
  RUAN: { initial: 'l', medial: 'u', rime: 'an', fuzzyHash: 'luan' },
  DONG: { initial: 'd', medial: 'u', rime: 'on', fuzzyHash: 'don' },
  TANG: { initial: 't', medial: '', rime: 'an', fuzzyHash: 'tan' },
  LAN: { initial: 'l', medial: '', rime: 'an', fuzzyHash: 'lan' },
  JIANG: { initial: 'j', medial: 'i', rime: 'an', fuzzyHash: 'jian' },
  SHAO: { initial: 's', medial: '', rime: 'ao', fuzzyHash: 'sao' },
  KANG: { initial: 'k', medial: '', rime: 'an', fuzzyHash: 'kan' },
  QIU: { initial: 'q', medial: 'i', rime: 'ou', fuzzyHash: 'qiou' },
  CHU: { initial: 'c', medial: 'u', rime: 'u', fuzzyHash: 'cu' },
  WEI: { initial: 'w', medial: 'u', rime: 'ei', fuzzyHash: 'wei' },
  YUAN: { initial: 'y', medial: 'u', rime: 'an', fuzzyHash: 'yuan' },
  YU: { initial: 'y', medial: 'u', rime: 'u', fuzzyHash: 'yu' },
  KE: { initial: 'k', medial: '', rime: 'e', fuzzyHash: 'ke' },
  ZHU: { initial: 'z', medial: 'u', rime: 'u', fuzzyHash: 'zu' },
  REN: { initial: 'l', medial: '', rime: 'en', fuzzyHash: 'len' },
  MI: { initial: 'm', medial: 'i', rime: 'i', fuzzyHash: 'mi' },
  SHEN: { initial: 's', medial: '', rime: 'en', fuzzyHash: 'sen' },
  RU: { initial: 'l', medial: 'u', rime: 'u', fuzzyHash: 'lu' },
  ZONG: { initial: 'z', medial: 'u', rime: 'on', fuzzyHash: 'zon' },
  GONG: { initial: 'g', medial: 'u', rime: 'on', fuzzyHash: 'gon' },
  XIANG: { initial: 'x', medial: 'i', rime: 'an', fuzzyHash: 'xian' },
  MA: { initial: 'm', medial: '', rime: 'a', fuzzyHash: 'ma' },
  JING: { initial: 'j', medial: 'i', rime: 'in', fuzzyHash: 'jin' },
  JUN: { initial: 'j', medial: 'u', rime: 'un', fuzzyHash: 'jun' },
  YING_YI: { initial: 'y', medial: 'i', rime: 'i', fuzzyHash: 'yi' },
  BO: { initial: 'b', medial: '', rime: 'o', fuzzyHash: 'bo' },
  JIE: { initial: 'j', medial: 'i', rime: 'ie', fuzzyHash: 'jie' },
  SHI: { initial: 's', medial: '', rime: 'i', fuzzyHash: 'si' },
  JIAO: { initial: 'j', medial: 'i', rime: 'ao', fuzzyHash: 'jiao' },
  SHOU: { initial: 's', medial: '', rime: 'ou', fuzzyHash: 'sou' },
  DIAN: { initial: 'd', medial: 'i', rime: 'an', fuzzyHash: 'dian' },
  MIAN: { initial: 'm', medial: 'i', rime: 'an', fuzzyHash: 'mian' },
  FAN: { initial: 'f', medial: '', rime: 'an', fuzzyHash: 'fan' },
  XIU: { initial: 'x', medial: 'i', rime: 'ou', fuzzyHash: 'xiou' },
  KAI: { initial: 'k', medial: '', rime: 'ai', fuzzyHash: 'kai' },
  XIN: { initial: 'x', medial: 'i', rime: 'in', fuzzyHash: 'xin' },
  BIAO: { initial: 'b', medial: 'i', rime: 'ao', fuzzyHash: 'biao' },
  BU: { initial: 'b', medial: 'u', rime: 'u', fuzzyHash: 'bu' },
  YAO: { initial: 'y', medial: 'i', rime: 'ao', fuzzyHash: 'yao' },
  LANG: { initial: 'l', medial: '', rime: 'an', fuzzyHash: 'lan' },
  DAI: { initial: 'd', medial: '', rime: 'ai', fuzzyHash: 'dai' },
  JIAN: { initial: 'j', medial: 'i', rime: 'an', fuzzyHash: 'jian' },
  QUN: { initial: 'q', medial: 'u', rime: 'un', fuzzyHash: 'qun' },
  FU: { initial: 'f', medial: 'u', rime: 'u', fuzzyHash: 'fu' },
  DAO: { initial: 'd', medial: '', rime: 'ao', fuzzyHash: 'dao' },
  QIE: { initial: 'q', medial: 'i', rime: 'ie', fuzzyHash: 'qie' },
  XIAO: { initial: 'x', medial: 'i', rime: 'ao', fuzzyHash: 'xiao' },
  XIAO_2: { initial: 'x', medial: 'i', rime: 'ao', fuzzyHash: 'xiao' },
  QI: { initial: 'q', medial: 'i', rime: 'i', fuzzyHash: 'qi' },
  ZHI: { initial: 'z', medial: '', rime: 'i', fuzzyHash: 'zi' },
  XI: { initial: 'x', medial: 'i', rime: 'i', fuzzyHash: 'xi' },
  JIU: { initial: 'j', medial: 'i', rime: 'ou', fuzzyHash: 'jiou' },
  JIA: { initial: 'j', medial: 'i', rime: 'a', fuzzyHash: 'jia' },
  MENG: { initial: 'm', medial: '', rime: 'en', fuzzyHash: 'men' },
  YI: { initial: 'y', medial: 'i', rime: 'i', fuzzyHash: 'yi' },
  PING: { initial: 'p', medial: 'i', rime: 'in', fuzzyHash: 'pin' },
  XUAN: { initial: 'x', medial: 'u', rime: 'an', fuzzyHash: 'xuan' },
  HE: { initial: 'h', medial: '', rime: 'e', fuzzyHash: 'he' },
  GEI_BEI: { initial: 'g', medial: '', rime: 'ei', fuzzyHash: 'gei' },
  ZHUANG: { initial: 'z', medial: 'u', rime: 'an', fuzzyHash: 'zuan' },
  CHONG: { initial: 'c', medial: 'u', rime: 'on', fuzzyHash: 'con' },
  ZU_FOOT: { initial: 'z', medial: 'u', rime: 'u', fuzzyHash: 'zu' },
  ZEN: { initial: 'z', medial: '', rime: 'en', fuzzyHash: 'zen' },
  ME_PARTICLE: { initial: 'm', medial: '', rime: 'e', fuzzyHash: 'me' },
  BAN: { initial: 'b', medial: '', rime: 'an', fuzzyHash: 'ban' },
  FENG: { initial: 'f', medial: 'u', rime: 'on', fuzzyHash: 'fon' },
  KU: { initial: 'k', medial: 'u', rime: 'u', fuzzyHash: 'ku' },
  YOU: { initial: 'y', medial: 'i', rime: 'ou', fuzzyHash: 'you' },
  MIN: { initial: 'm', medial: 'i', rime: 'in', fuzzyHash: 'min' },
  HUI: { initial: 'h', medial: 'u', rime: 'ei', fuzzyHash: 'hui' },
  NA: { initial: 'n', medial: '', rime: 'a', fuzzyHash: 'na' },
  AN: { initial: 'y', medial: '', rime: 'an', fuzzyHash: 'an' },
  PEI: { initial: 'p', medial: '', rime: 'ei', fuzzyHash: 'pei' },
  XUN: { initial: 'x', medial: 'u', rime: 'un', fuzzyHash: 'xun' },
  DIAO: { initial: 'd', medial: 'i', rime: 'ao', fuzzyHash: 'diao' },
  QIAO: { initial: 'q', medial: 'i', rime: 'ao', fuzzyHash: 'qiao' },
  BA: { initial: 'b', medial: '', rime: 'a', fuzzyHash: 'ba' },
  LIAN: { initial: 'l', medial: 'i', rime: 'an', fuzzyHash: 'lian' },
  ZHUAN: { initial: 'z', medial: 'u', rime: 'an', fuzzyHash: 'zuan' },
  DAN: { initial: 'd', medial: '', rime: 'an', fuzzyHash: 'dan' },
  DU: { initial: 'd', medial: 'u', rime: 'u', fuzzyHash: 'du' },
  YUE: { initial: 'y', medial: 'u', rime: 'ie', fuzzyHash: 'yue' },
  HAO: { initial: 'h', medial: '', rime: 'ao', fuzzyHash: 'hao' },
  ZAO: { initial: 'z', medial: '', rime: 'ao', fuzzyHash: 'zao' },
  SHANG: { initial: 's', medial: '', rime: 'an', fuzzyHash: 'san' },
  TIAN: { initial: 't', medial: 'i', rime: 'an', fuzzyHash: 'tian' },
  SAN: { initial: 's', medial: '', rime: 'an', fuzzyHash: 'san' },
  LIANG: { initial: 'l', medial: 'i', rime: 'an', fuzzyHash: 'lian' },

  // 擴充全音節映射
  A: { initial: 'y', medial: '', rime: 'a', fuzzyHash: 'a' },
  AI: { initial: 'y', medial: '', rime: 'ai', fuzzyHash: 'ai' },
  AO: { initial: 'y', medial: '', rime: 'ao', fuzzyHash: 'ao' },
  OU: { initial: 'y', medial: '', rime: 'ou', fuzzyHash: 'ou' },
  EN: { initial: 'y', medial: '', rime: 'en', fuzzyHash: 'en' },
  ER: { initial: 'l', medial: '', rime: 'e', fuzzyHash: 'er' },
  PI: { initial: 'p', medial: 'i', rime: 'i', fuzzyHash: 'pi' },
  PU: { initial: 'p', medial: 'u', rime: 'u', fuzzyHash: 'pu' },
  PA: { initial: 'p', medial: '', rime: 'a', fuzzyHash: 'pa' },
  PAI: { initial: 'p', medial: '', rime: 'ai', fuzzyHash: 'pai' },
  PAO: { initial: 'p', medial: '', rime: 'ao', fuzzyHash: 'pao' },
  PEN: { initial: 'p', medial: '', rime: 'en', fuzzyHash: 'pen' },
  PIAO: { initial: 'p', medial: 'i', rime: 'ao', fuzzyHash: 'piao' },
  PIAN: { initial: 'p', medial: 'i', rime: 'an', fuzzyHash: 'pian' },
  PIE: { initial: 'p', medial: 'i', rime: 'ie', fuzzyHash: 'pie' },
  MO: { initial: 'm', medial: '', rime: 'o', fuzzyHash: 'mo' },
  MAI: { initial: 'm', medial: '', rime: 'ai', fuzzyHash: 'mai' },
  MAO: { initial: 'm', medial: '', rime: 'ao', fuzzyHash: 'mao' },
  MU: { initial: 'm', medial: 'u', rime: 'u', fuzzyHash: 'mu' },
  MEI: { initial: 'm', medial: '', rime: 'ei', fuzzyHash: 'mei' },
  MEN: { initial: 'm', medial: '', rime: 'en', fuzzyHash: 'men' },
  MAN: { initial: 'm', medial: '', rime: 'an', fuzzyHash: 'man' },
  MIAO: { initial: 'm', medial: 'i', rime: 'ao', fuzzyHash: 'miao' },
  MIE: { initial: 'm', medial: 'i', rime: 'ie', fuzzyHash: 'mie' },
  MIU: { initial: 'm', medial: 'i', rime: 'ou', fuzzyHash: 'miou' },
  FA_HUA: { initial: 'f', medial: 'u', rime: 'a', fuzzyHash: 'fa' },
  FEI_HUI: { initial: 'f', medial: 'u', rime: 'ei', fuzzyHash: 'fei' },
  FEN_HUN: { initial: 'f', medial: 'u', rime: 'en', fuzzyHash: 'fen' },
  FO: { initial: 'f', medial: 'u', rime: 'o', fuzzyHash: 'fo' },
  FOU: { initial: 'f', medial: '', rime: 'ou', fuzzyHash: 'fou' },
  DA_TA: { initial: 'd', medial: '', rime: 'a', fuzzyHash: 'da' },
  DE_DI: { initial: 'd', medial: 'i', rime: 'i', fuzzyHash: 'di' },
  DAO_TAO: { initial: 'd', medial: '', rime: 'ao', fuzzyHash: 'dao' },
  DOU: { initial: 'd', medial: '', rime: 'ou', fuzzyHash: 'dou' },
  DUI_TUI: { initial: 'd', medial: 'u', rime: 'ei', fuzzyHash: 'dui' },
  DUAN_TUAN: { initial: 'd', medial: 'u', rime: 'an', fuzzyHash: 'duan' },
  DUN_TUN: { initial: 'd', medial: 'u', rime: 'un', fuzzyHash: 'dun' },
  DUO_TUO: { initial: 'd', medial: 'u', rime: 'o', fuzzyHash: 'duo' },
  DIE_TIE: { initial: 'd', medial: 'i', rime: 'ie', fuzzyHash: 'die' },
  DIU: { initial: 'd', medial: 'i', rime: 'ou', fuzzyHash: 'diou' },
  DONG_TONG: { initial: 'd', medial: 'u', rime: 'on', fuzzyHash: 'don' },
  TAI_TAN: { initial: 't', medial: '', rime: 'an', fuzzyHash: 'tan' },
  TE_CE: { initial: 'c', medial: '', rime: 'e', fuzzyHash: 'ce' },
  TI: { initial: 't', medial: 'i', rime: 'i', fuzzyHash: 'ti' },
  TIAO: { initial: 't', medial: 'i', rime: 'ao', fuzzyHash: 'tiao' },
  TING: { initial: 't', medial: 'i', rime: 'in', fuzzyHash: 'tin' },
  NA_LA: { initial: 'l', medial: '', rime: 'a', fuzzyHash: 'la' },
  NAI_LAI: { initial: 'l', medial: '', rime: 'ai', fuzzyHash: 'lai' },
  NAN_LAN: { initial: 'l', medial: '', rime: 'an', fuzzyHash: 'lan' },
  NANG_LANG: { initial: 'l', medial: '', rime: 'an', fuzzyHash: 'lan' },
  NAO_LAO: { initial: 'l', medial: '', rime: 'ao', fuzzyHash: 'lao' },
  NEI_LEI: { initial: 'l', medial: '', rime: 'ei', fuzzyHash: 'lei' },
  NEN_LENG: { initial: 'l', medial: '', rime: 'en', fuzzyHash: 'len' },
  NI_LI: { initial: 'l', medial: 'i', rime: 'i', fuzzyHash: 'li' },
  NIAN_LIAN: { initial: 'l', medial: 'i', rime: 'an', fuzzyHash: 'lian' },
  NIANG_LIANG: { initial: 'l', medial: 'i', rime: 'an', fuzzyHash: 'lian' },
  NIAO_LIE: { initial: 'l', medial: 'i', rime: 'ie', fuzzyHash: 'lie' },
  NIN_LING: { initial: 'l', medial: 'i', rime: 'in', fuzzyHash: 'lin' },
  NIU_LIU: { initial: 'l', medial: 'i', rime: 'ou', fuzzyHash: 'liou' },
  NONG_LONG: { initial: 'l', medial: 'u', rime: 'on', fuzzyHash: 'lon' },
  NU_LYU: { initial: 'l', medial: 'u', rime: 'u', fuzzyHash: 'lu' },
  NUAN_LUAN: { initial: 'l', medial: 'u', rime: 'an', fuzzyHash: 'luan' },
  NYUE_LYUE: { initial: 'l', medial: 'u', rime: 'ie', fuzzyHash: 'lue' },
  GE_KE: { initial: 'g', medial: '', rime: 'e', fuzzyHash: 'ge' },
  GAI_KAI: { initial: 'g', medial: '', rime: 'ai', fuzzyHash: 'gai' },
  GAN_KAN: { initial: 'g', medial: '', rime: 'an', fuzzyHash: 'gan' },
  GANG_KANG: { initial: 'g', medial: '', rime: 'an', fuzzyHash: 'gan' },
  GAO_KAO: { initial: 'g', medial: '', rime: 'ao', fuzzyHash: 'gao' },
  GOU_KOU: { initial: 'g', medial: '', rime: 'ou', fuzzyHash: 'gou' },
  GU_KU: { initial: 'g', medial: 'u', rime: 'u', fuzzyHash: 'gu' },
  GUA_KUA: { initial: 'g', medial: 'u', rime: 'a', fuzzyHash: 'gua' },
  GUAI_KUAI: { initial: 'g', medial: 'u', rime: 'ai', fuzzyHash: 'guai' },
  GUAN_KUAN: { initial: 'g', medial: 'u', rime: 'an', fuzzyHash: 'guan' },
  GUANG_KUANG: { initial: 'g', medial: 'u', rime: 'an', fuzzyHash: 'guan' },
  GUI_KUI: { initial: 'g', medial: 'u', rime: 'ei', fuzzyHash: 'gui' },
  GUN_KUN: { initial: 'g', medial: 'u', rime: 'un', fuzzyHash: 'gun' },
  HA_HE: { initial: 'h', medial: '', rime: 'e', fuzzyHash: 'he' },
  HAI: { initial: 'h', medial: '', rime: 'ai', fuzzyHash: 'hai' },
  HAN: { initial: 'h', medial: '', rime: 'an', fuzzyHash: 'han' },
  HANG: { initial: 'h', medial: '', rime: 'an', fuzzyHash: 'han' },
  HEI_HEN: { initial: 'h', medial: '', rime: 'en', fuzzyHash: 'hen' },
  HENG_HONG: { initial: 'h', medial: 'u', rime: 'on', fuzzyHash: 'hon' },
  HU_FU: { initial: 'h', medial: 'u', rime: 'u', fuzzyHash: 'hu' },
  HUAI: { initial: 'h', medial: 'u', rime: 'ai', fuzzyHash: 'huai' },
  HUAN_HUANG: { initial: 'h', medial: 'u', rime: 'an', fuzzyHash: 'huan' },
  HUN: { initial: 'h', medial: 'u', rime: 'un', fuzzyHash: 'hun' },
  JI: { initial: 'j', medial: 'i', rime: 'i', fuzzyHash: 'ji' },
  QIA: { initial: 'q', medial: 'i', rime: 'a', fuzzyHash: 'qia' },
  QIANG: { initial: 'q', medial: 'i', rime: 'an', fuzzyHash: 'qian' },
  QIN_QING: { initial: 'q', medial: 'i', rime: 'in', fuzzyHash: 'qin' },
  QIONG: { initial: 'q', medial: 'u', rime: 'on', fuzzyHash: 'qion' },
  QU: { initial: 'q', medial: 'u', rime: 'u', fuzzyHash: 'qu' },
  QUAN: { initial: 'q', medial: 'u', rime: 'an', fuzzyHash: 'quan' },
  QUE: { initial: 'q', medial: 'u', rime: 'ie', fuzzyHash: 'que' },
  XIA: { initial: 'x', medial: 'i', rime: 'a', fuzzyHash: 'xia' },
  XIONG: { initial: 'x', medial: 'u', rime: 'on', fuzzyHash: 'xion' },
  ZHA_CA: { initial: 'z', medial: '', rime: 'a', fuzzyHash: 'za' },
  ZHAI_CAI: { initial: 'z', medial: '', rime: 'ai', fuzzyHash: 'zai' },
  ZHAO_CHAO: { initial: 'z', medial: '', rime: 'ao', fuzzyHash: 'zao' },
  ZHE_CHE: { initial: 'z', medial: '', rime: 'e', fuzzyHash: 'ze' },
  ZHUO_CUO: { initial: 'z', medial: 'u', rime: 'o', fuzzyHash: 'zuo' },
  ZHUI_CUI: { initial: 'z', medial: 'u', rime: 'ei', fuzzyHash: 'zui' },
  ZHUN_CUN: { initial: 'z', medial: 'u', rime: 'un', fuzzyHash: 'zun' },
  ZI_CI: { initial: 'z', medial: '', rime: 'i', fuzzyHash: 'zi' },
  ZU_CU: { initial: 'z', medial: 'u', rime: 'u', fuzzyHash: 'zu' },
  ZUAN_CUAN: { initial: 'z', medial: 'u', rime: 'an', fuzzyHash: 'zuan' },
  ZUI: { initial: 'z', medial: 'u', rime: 'ei', fuzzyHash: 'zui' },
  ZUO: { initial: 'z', medial: 'u', rime: 'o', fuzzyHash: 'zuo' },
  SHA_SA: { initial: 's', medial: '', rime: 'a', fuzzyHash: 'sa' },
  SHAI_SAI: { initial: 's', medial: '', rime: 'ai', fuzzyHash: 'sai' },
  SHAO_SAO: { initial: 's', medial: '', rime: 'ao', fuzzyHash: 'sao' },
  SHE_SE: { initial: 's', medial: '', rime: 'e', fuzzyHash: 'se' },
  SHUI_SUI: { initial: 's', medial: 'u', rime: 'ei', fuzzyHash: 'sui' },
  SHUA_SHUAI: { initial: 's', medial: 'u', rime: 'ai', fuzzyHash: 'suai' },
  SHUAN_SUAN: { initial: 's', medial: 'u', rime: 'an', fuzzyHash: 'suan' },
  SHUO_SUO: { initial: 's', medial: 'u', rime: 'o', fuzzyHash: 'suo' },
  SI_SHI: { initial: 's', medial: '', rime: 'i', fuzzyHash: 'si' },
  SONG_SHU: { initial: 's', medial: 'u', rime: 'u', fuzzyHash: 'su' },
  SOU: { initial: 's', medial: '', rime: 'ou', fuzzyHash: 'sou' },
  RAN: { initial: 'l', medial: '', rime: 'an', fuzzyHash: 'lan' },
  RANG: { initial: 'l', medial: '', rime: 'an', fuzzyHash: 'lan' },
  RAO: { initial: 'l', medial: '', rime: 'ao', fuzzyHash: 'lao' },
  RE_LE: { initial: 'l', medial: '', rime: 'e', fuzzyHash: 'le' },
  RENG: { initial: 'l', medial: '', rime: 'en', fuzzyHash: 'len' },
  RI: { initial: 'l', medial: '', rime: 'i', fuzzyHash: 'li' },
  RONG: { initial: 'l', medial: 'u', rime: 'on', fuzzyHash: 'lon' },
  ROU: { initial: 'l', medial: '', rime: 'ou', fuzzyHash: 'lou' },
  RUI: { initial: 'l', medial: 'u', rime: 'ei', fuzzyHash: 'lui' },
  RUN: { initial: 'l', medial: 'u', rime: 'un', fuzzyHash: 'lun' },
  RUO: { initial: 'l', medial: 'u', rime: 'o', fuzzyHash: 'luo' },
  YA: { initial: 'y', medial: 'i', rime: 'a', fuzzyHash: 'ya' },
  YOU_2: { initial: 'y', medial: 'i', rime: 'ou', fuzzyHash: 'you' },
  YONG: { initial: 'y', medial: 'u', rime: 'on', fuzzyHash: 'yon' },
  WA: { initial: 'w', medial: 'u', rime: 'a', fuzzyHash: 'wa' },
  WAI: { initial: 'w', medial: 'u', rime: 'ai', fuzzyHash: 'wai' },
  WO: { initial: 'w', medial: 'u', rime: 'o', fuzzyHash: 'wo' },
};

/**
 * 台灣口語會議常見雙音節連音縮讀表 (Syllable Contraction / Rapid Speech Sandhi)
 * 例如：「這樣」->「醬/降」、「不要」->「表」、「知道」->「造」、「什麼」->「啥」
 */
const COLLOQUIAL_CONTRACTION_PAIRS: Array<[string, string]> = [
  ['這樣', '醬'],
  ['這樣', '降'],
  ['那樣', '釀'],
  ['不要', '表'],
  ['不要', '甭'],
  ['知道', '造'],
  ['什麼', '啥'],
  ['怎麼', '怎'],
  ['沒有', '沒'],
  ['我們', '們'],
  ['你們', '您'],
  ['這個', '這'],
  ['那個', '那'],
  ['禮拜', '拜'],
  ['二十', '廿'],
  ['三十', '卅'],
];

function charToSyllableTuple(ch: string): SyllablePhonemeTuple {
  const code = charToPhoneticCode.get(ch);
  if (code && CODE_TO_TUPLE_MAP[code]) {
    const t = CODE_TO_TUPLE_MAP[code];
    return {
      rawCode: code,
      fuzzyHash: t.fuzzyHash,
      initial: t.initial,
      medial: t.medial,
      rime: t.rime,
    };
  }
  return {
    rawCode: ch,
    fuzzyHash: ch.toLowerCase(),
    initial: ch,
    medial: '',
    rime: ch,
  };
}

export function toPhoneticSignature(text: string): string {
  return Array.from(text)
    .map((ch) => charToPhoneticCode.get(ch) || ch)
    .join('-');
}

/**
 * 取得台式華語模糊音韻序列 (供 UncertainCandidateOption.pinyinSequence 與雙向重疊消融使用)
 */
export function toTaiwanFuzzyPhoneticSyllables(text: string): string[] {
  return Array.from(text).map((ch) => charToSyllableTuple(ch).fuzzyHash);
}

/**
 * 計算兩個單音節之台式腔調與快速連讀替換成本 c_sub(s_i, t_j) \in [0, 1]
 */
function computeSyllableSubCost(
  a: SyllablePhonemeTuple,
  b: SyllablePhonemeTuple
): number {
  if (a.fuzzyHash === b.fuzzyHash) return 0.0;

  // 高頻台灣國語近音對轉 (平捲舌 + 圓展唇或鼻化弱化)
  if (
    (a.fuzzyHash === 'su' && b.fuzzyHash === 'si') ||
    (a.fuzzyHash === 'si' && b.fuzzyHash === 'su') ||
    (a.fuzzyHash === 'si' && b.fuzzyHash === 'zi') ||
    (a.fuzzyHash === 'zi' && b.fuzzyHash === 'si') ||
    (a.fuzzyHash === 'zu' && b.fuzzyHash === 'cu') ||
    (a.fuzzyHash === 'cu' && b.fuzzyHash === 'zu') ||
    (a.fuzzyHash === 'fa' && b.fuzzyHash === 'wa') ||
    (a.fuzzyHash === 'wa' && b.fuzzyHash === 'fa') ||
    (a.fuzzyHash === 'fei' && b.fuzzyHash === 'hui') ||
    (a.fuzzyHash === 'hui' && b.fuzzyHash === 'fei') ||
    (a.fuzzyHash === 'fen' && b.fuzzyHash === 'hun') ||
    (a.fuzzyHash === 'hun' && b.fuzzyHash === 'fen') ||
    (a.fuzzyHash === 'hu' && b.fuzzyHash === 'fu') ||
    (a.fuzzyHash === 'fu' && b.fuzzyHash === 'hu')
  ) {
    return 0.12;
  }
  if (
    (a.fuzzyHash === 'zu' && b.fuzzyHash === 'li') ||
    (a.fuzzyHash === 'li' && b.fuzzyHash === 'zu') ||
    (a.fuzzyHash === 'lu' && b.fuzzyHash === 'li') ||
    (a.fuzzyHash === 'li' && b.fuzzyHash === 'lu') ||
    (a.fuzzyHash === 'fon' && b.fuzzyHash === 'gon') ||
    (a.fuzzyHash === 'gon' && b.fuzzyHash === 'fon') ||
    (a.fuzzyHash === 'fon' && b.fuzzyHash === 'hon') ||
    (a.fuzzyHash === 'hon' && b.fuzzyHash === 'fon') ||
    (a.fuzzyHash === 'jian' && b.fuzzyHash === 'qian') ||
    (a.fuzzyHash === 'qian' && b.fuzzyHash === 'jian') ||
    (a.fuzzyHash === 'jin' && b.fuzzyHash === 'qin') ||
    (a.fuzzyHash === 'qin' && b.fuzzyHash === 'jin')
  ) {
    return 0.16;
  }
  if (
    (a.fuzzyHash === 'wu' && b.fuzzyHash === 'guo') ||
    (a.fuzzyHash === 'guo' && b.fuzzyHash === 'wu') ||
    (a.fuzzyHash === 'wei' && b.fuzzyHash === 'wan') ||
    (a.fuzzyHash === 'wan' && b.fuzzyHash === 'wei') ||
    (a.fuzzyHash === 'yuan' && b.fuzzyHash === 'yu') ||
    (a.fuzzyHash === 'yu' && b.fuzzyHash === 'yuan') ||
    (a.fuzzyHash === 'zuan' && b.fuzzyHash === 'con') ||
    (a.fuzzyHash === 'con' && b.fuzzyHash === 'zuan') ||
    (a.fuzzyHash === 'xiou' && b.fuzzyHash === 'zu') ||
    (a.fuzzyHash === 'zu' && b.fuzzyHash === 'xiou') ||
    (a.fuzzyHash === 'na' && b.fuzzyHash === 'dai') ||
    (a.fuzzyHash === 'dai' && b.fuzzyHash === 'na') ||
    (a.fuzzyHash === 'an' && b.fuzzyHash === 'ban') ||
    (a.fuzzyHash === 'ban' && b.fuzzyHash === 'an')
  ) {
    return 0.20;
  }

  let initialCost = 1.0;
  if (a.initial === b.initial) {
    initialCost = 0.0;
  } else if (
    (a.initial === 'b' && b.initial === 'p') ||
    (a.initial === 'p' && b.initial === 'b') ||
    (a.initial === 'd' && b.initial === 't') ||
    (a.initial === 't' && b.initial === 'd') ||
    (a.initial === 'g' && b.initial === 'k') ||
    (a.initial === 'k' && b.initial === 'g') ||
    (['l', 'r', 'n', 'y', 'd'].includes(a.initial) &&
      ['l', 'r', 'n', 'y', 'd'].includes(b.initial)) ||
    (['w', 'g', 'h', 'f'].includes(a.initial) &&
      ['w', 'g', 'h', 'f'].includes(b.initial) &&
      (a.medial === 'u' || b.medial === 'u')) ||
    (['j', 'q', 'x'].includes(a.initial) &&
      ['j', 'q', 'x'].includes(b.initial)) ||
    (['z', 'c', 's'].includes(a.initial) &&
      ['z', 'c', 's'].includes(b.initial)) ||
    (['j', 'q', 'x', 'z', 'c', 's'].includes(a.initial) &&
      ['j', 'q', 'x', 'z', 'c', 's'].includes(b.initial))
  ) {
    initialCost = 0.22;
  }

  let rimeCost = 1.0;
  if (a.rime === b.rime) {
    rimeCost = 0.0;
  } else if (
    (a.rime === 'an' && b.rime === 'en') ||
    (a.rime === 'en' && b.rime === 'an') ||
    (a.rime === 'in' && b.rime === 'un') ||
    (a.rime === 'un' && b.rime === 'in') ||
    (a.rime === 'in' && b.rime === 'i') ||
    (a.rime === 'i' && b.rime === 'in') ||
    (a.rime === 'u' && b.rime === 'o') ||
    (a.rime === 'o' && b.rime === 'u') ||
    (a.rime === 'ei' && b.rime === 'an') ||
    (a.rime === 'an' && b.rime === 'ei') ||
    (a.rime === 'i' && b.rime === 'u') ||
    (a.rime === 'u' && b.rime === 'i') ||
    (a.rime === 'ie' && b.rime === 'i') ||
    (a.rime === 'i' && b.rime === 'ie') ||
    (a.rime === 'ai' && b.rime === 'a') ||
    (a.rime === 'a' && b.rime === 'ai') ||
    (a.rime === 'on' && b.rime === 'en') ||
    (a.rime === 'en' && b.rime === 'on')
  ) {
    rimeCost = 0.20;
  }

  const medialCost = a.medial === b.medial ? 0.0 : 0.16;
  return Math.min(1.0, 0.45 * initialCost + 0.45 * rimeCost + 0.1 * medialCost);
}

/**
 * 判斷雙音節 (s_{i-1}, s_i) 是否可經由快速連音縮讀 (Syllable Contraction 2 -> 1) 合併為單音節 t_j
 */
function canContractTwoSyllablesIntoOne(
  sPrev: SyllablePhonemeTuple,
  sCurr: SyllablePhonemeTuple,
  target: SyllablePhonemeTuple
): boolean {
  // 檢查是否符合台灣口語高頻合音規則
  for (const [twoSyl, oneSyl] of COLLOQUIAL_CONTRACTION_PAIRS) {
    const t0 = charToSyllableTuple(twoSyl[0]);
    const t1 = charToSyllableTuple(twoSyl[1]);
    const tc = charToSyllableTuple(oneSyl[0]);
    if (
      sPrev.fuzzyHash === t0.fuzzyHash &&
      sCurr.fuzzyHash === t1.fuzzyHash &&
      target.fuzzyHash === tc.fuzzyHash
    ) {
      return true;
    }
  }

  if (sPrev.initial !== target.initial && sCurr.initial !== target.initial) {
    return false;
  }
  const secondHasYOrI =
    sCurr.initial === 'y' || sCurr.medial === 'i' || sCurr.rime === target.rime;
  return secondHasYOrI && (sCurr.rime === target.rime || sPrev.rime === target.rime);
}

/**
 * 支援連音縮讀與標點忽略之動態規劃加權音韻編輯距離 d_phon(S, T) \in [0, 1]
 */
export function computeTaiwanPhoneticDistance(
  sourceText: string,
  targetText: string
): number {
  const cleanS = sourceText.replace(/[，。？！、；：,.?!\s]/g, '').trim();
  const cleanT = targetText.replace(/[，。？！、；：,.?!\s]/g, '').trim();
  const sChars = Array.from(cleanS);
  const tChars = Array.from(cleanT);
  const n = sChars.length;
  const m = tChars.length;
  if (n === 0 && m === 0) return 0.0;
  if (n === 0 || m === 0) return 1.0;
  if (cleanS === cleanT) return 0.0;

  const sTuples = sChars.map(charToSyllableTuple);
  const tTuples = tChars.map(charToSyllableTuple);

  const indelCost = 0.72;
  const contractCost = 0.20;

  const dp: number[][] = Array.from({ length: n + 1 }, () =>
    new Array(m + 1).fill(Infinity)
  );
  dp[0][0] = 0.0;

  for (let i = 1; i <= n; i++) dp[i][0] = i * indelCost;
  for (let j = 1; j <= m; j++) dp[0][j] = j * indelCost;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const sub =
        dp[i - 1][j - 1] + computeSyllableSubCost(sTuples[i - 1], tTuples[j - 1]);
      const del = dp[i - 1][j] + indelCost;
      const ins = dp[i][j - 1] + indelCost;
      let best = Math.min(sub, del, ins);

      if (
        i >= 2 &&
        canContractTwoSyllablesIntoOne(
          sTuples[i - 2],
          sTuples[i - 1],
          tTuples[j - 1]
        )
      ) {
        best = Math.min(best, dp[i - 2][j - 1] + contractCost);
      }

      if (
        j >= 2 &&
        canContractTwoSyllablesIntoOne(
          tTuples[j - 2],
          tTuples[j - 1],
          sTuples[i - 1]
        )
      ) {
        best = Math.min(best, dp[i - 1][j - 2] + contractCost);
      }

      dp[i][j] = best;
    }
  }

  const normDenom = Math.max(n, m);
  return Number(Math.min(1.0, dp[n][m] / normDenom).toFixed(3));
}

/**
 * 當使用者更名語者（如 Speaker_01 -> 林處長）時，掃描全文找出同音誤植詞（如「林築長」、「臨處長」）
 * 支援單一 WordSpan 或跨鄰近 WordSpan 之同音偵測
 */
export function scanHomophoneSpeakerMentions(
  targetSpeakerName: string,
  segments: Segment[]
): PhoneticMatchCandidate[] {
  const cleanTarget = targetSpeakerName.trim();
  if (cleanTarget.length < 2) return [];

  const targetSig = toPhoneticSignature(cleanTarget);
  const candidates: PhoneticMatchCandidate[] = [];

  for (const seg of segments) {
    for (let i = 0; i < seg.words.length; i++) {
      const wordText = seg.words[i].text;
      if (wordText.length === cleanTarget.length && wordText !== cleanTarget) {
        if (
          toPhoneticSignature(wordText) === targetSig ||
          computeTaiwanPhoneticDistance(wordText, cleanTarget) <= 0.18
        ) {
          const prev = seg.words[i - 1]?.text || '';
          const next = seg.words[i + 1]?.text || '';
          candidates.push({
            segmentId: seg.id,
            wordIndex: i,
            matchedText: wordText,
            targetSpeakerName: cleanTarget,
            contextPreview: `...${prev}【${wordText}】${next}...`,
            selected: true,
          });
        }
      }
    }
  }
  return candidates;
}

export const computePhoneticSignature = toPhoneticSignature;
export const scanTranscriptForPhoneticMatches = scanHomophoneSpeakerMentions;
export const findPhoneticCandidates = scanHomophoneSpeakerMentions;

export function toAccentInsensitivePhoneticKey(text: string): string {
  const syls = toTaiwanFuzzyPhoneticSyllables(text);
  return syls.join('-');
}

/**
 * 結合別名精確替換與滑動視窗音韻距離匹配之領域詞典自動校正
 * 1. 先執行已知 aliases 精確匹配替換
 * 2. 再針對 3 字以上專有名詞執行滑動視窗近音校正 (d_phon <= 0.14)，避免漏網之同音錯字
 */
export function applyPhoneticDictionaryCorrection(
  text: string,
  dictionary: Array<{ term: string; aliases?: string[] }>
): string {
  let out = text;
  for (const d of dictionary) {
    if (d.aliases) {
      for (const a of d.aliases) {
        if (a && out.includes(a)) {
          out = out.split(a).join(d.term);
        }
      }
    }
  }

  // 對 3 字以上高價值專有名詞進行滑動視窗模糊同音替換
  for (const d of dictionary) {
    const term = d.term?.trim();
    if (!term || term.length < 3 || !/^[\u4e00-\u9fa5]+$/.test(term)) continue;
    const tLen = term.length;
    const termKey = toAccentInsensitivePhoneticKey(term);
    let idx = 0;
    while (idx <= out.length - tLen) {
      const sub = out.slice(idx, idx + tLen);
      if (sub !== term && /^[\u4e00-\u9fa5]+$/.test(sub)) {
        const subKey = toAccentInsensitivePhoneticKey(sub);
        if (
          subKey === termKey ||
          computeTaiwanPhoneticDistance(sub, term) <= 0.14
        ) {
          out = out.slice(0, idx) + term + out.slice(idx + tLen);
          idx += tLen;
          continue;
        }
      }
      idx++;
    }
  }

  return out;
}
