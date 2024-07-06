import json
from langchain_google_vertexai import VertexAI

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults


class OutputNameEntity(BaseModel):
    名前: str = Field(description="エンティティの名前またはヘッダーはコンテキスト内で抽出されます")


class GeminiHandler:
    def __init__(self):
        self.model_llm = VertexAI(model_name="gemini-1.0-pro-002", temperature=0.1)
        wrapper = DuckDuckGoSearchAPIWrapper(region="jp-jp", time="y", max_results=1)  # us-en  # jp-jp
        wrapper.region = "jp-jp"
        self.search_api = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")
        self.search_api.max_results = 5
        self.create_chain()

    def run_output_pl(self, profit_and_lost):
        return self.chain_pl.invoke(profit_and_lost)

    def final_output(self, output_of_ai_pl):
        list_entity = self.chain_get_name_entity.invoke({"context": output_of_ai_pl})
        # print("name_entity: ", list_entity)
        final_result = ""
        if len(list_entity) != 0:
            for ind, entity in enumerate(list_entity):
                classify = self.chain_classify.invoke(entity)
                if len(classify[0]) != 0:
                    summary_text = self.chain_summary.invoke({"snippet": classify[0]})
                    final_result += f"//***{list_entity[ind]}***//" + '\n'
                    final_result += summary_text + '\n'
                    final_result += classify[2] + "\n\n"
        return final_result

    def create_chain(self):
        self.create_chain_pl()
        self.create_chain_analytic()
        self.create_chain_get_name_entity()
        self.create_chain_check_relate()
        self.create_summary_snippet()
        self.create_chain_search()

    def create_chain_pl(self):
        template_pl = """
            以下に提供する情報を使用して経営分析・事業計画 [ 収益予測 ]を作成していただきたいです。 長年のエコノミストおよび市場調査者として、以下の要求された分析を提供するのはご協力ください。

            データ入力に基づいて、次の列を含む経営分析・事業計画 [ 収益予測 ]をテーブルで作成してください。テーブルには絶対に8列があり、そのうち最大3列が財務情報データ（実績）に基づく実績です（直近 3 年間のデータを表示して、3 年間のデータが不足している場合は、データのある年​​のみの列を表示してください）。 残りの 5 つの列は、今後 5 年間を絶対に予測して出力してください。
            ・財務数値の傾向を判断するには、財務情報の各月のデータを重線形回帰や移動平均や時系列解析で分析して、そこから企業理念（MVV）、経営方針、競合比較結果、3C分析結果、マーケティング戦略などの要素を組み合わせる基礎として使用します。そうすれば成長率を予測することができます。
            ・1 年は納税期間として認識され、4 月に始まり翌年 3 月に終わります。
            ・法人税等合計は税金等調整前当期純利益の 30,62 % に相当します。 税金は、次の期では表示されず、その会計期間で表示してください。
            ・当期純利益は税金等調整前当期純利益から法人税等合計を引いたものに等しいです。

            このテーブルは 3 つの連続した財務三表で構成されており、
            最初に以下のような行を持つ損益計算表が表示されます。
            次の見出しに従って 13 ロウを絶対に出力してください。
            「売上高」、「売上原価」、「売上総利益」、「販売費及び一般管理費」、「営業利益」、「営業外収入」、「営業外費用」、「経常利益」、「特別利益」、「特別損失」、「税金等調整前当期純利益」、「法人税等合計」、「当期純利益」

            損益計算書の下に予測の根拠を記載してください。 年平均成長率ではなく、毎年の成長率を詳しく説明してください。 % または説明は、実際には上記で指定した予測結果と正確に一致する必要があることに注意してください。

            ・列は垂直方向に表示する必要があり、行は水平方向に表示する必要があります。
            ・表の各項目を実績と予測される今後5年間ごとに出力してください。
            ・データ入力と計算された数値の単位は千円で、アウトプットはまた計算の段階があるので、そのまま出してください。
            ・予測して出力されたデータは必ず数字で出してください。
            ・予測の財務三表は相互に密接に関連する必要があります。


            注記：
            1. 私が要求した情報を除き、返された結果に質問や要求を繰り返さないでください。
            2. 回答してくれるのは絶対に日本語でお願いします。
            3. 私が要求したものの中に「絶対に」というのは、しなければならないことから、その通りちゃんと理解して実行してください。
            4. AIツールに任せるので、不足のデータを信頼できる研究にベースして、自動的に計画や成長数量などの予想を指摘してください。
            5. 抽出する観点は最新で挙げられている動向、記事の多さから優先的にピックアップする、鮮度の古い情報や誤情報をは外すること。

            情報：
            ・企業理念：社会課題解決に取り組む企業に“もっと明るい未来”を
            企業において心を動かすサステナビリティ（SDGs）・ESGを経営戦略に取り入れる時、そこには社会変革を興す“FORCE”が生まれる。政府・自治体・大手企業・銀行などがサステナビリティ・ESGへの取り組みを重視し始めている昨今、従来の財務と共に非財務の活動を推進することで、企業価値を最大化することに繋げる事ができる。
            Purpose：SX（サステナビリティ・トランスフォーメーション）の力で、世界中の次世代の若者や子ども達の100年後の未来を守る
            Mission：あらゆるステークホルダーが三歩先の未来を想像でき、希望を持てる社会創りに貢献する
            Vision：社会課題解決における「人」と「テクノロジー」の融合を促進する事でSX推進企業として世界No.1を目指す
            Values：尊重・誠意・謙虚・共感・創造・信頼
            「相互幸福」という価値を創出する

            ・経営方針：ESG 経営戦略における唯一無二のプラットフォームを目指します。まずは直近 2022 年～2024 年で中堅・中小企業 No.1 の ESG 運用ツール & サポートを目指し、長期的にはESG プラットフォーム & データカンパニーとして国際ベースのデータを基にした世界展開を目指します

            ・競合比較と差別化ポイント：
            | | 弊社 | 競合A（株式会社Drop） | 競合B（日本能率協会コンサルティング） | 競合C（Deloitte） |
            |---|---|---|---|---|
            | サービス名 | サステナビリティ経営支援サービス、ビジネススクール運営、コミュニティサービス提供 | ESG経営支援 | ESG経営支援 | ESG経営支援 |
            | 顧客対象 | 社会課題解決に取り組む企業 | 企業全般 | 企業全般 | 企業全般 |
            | 価格 | ◎（サブスク型、初期導入費用は別途必要） | △（詳細不明） | △（詳細不明） | ×（高額） |
            | 技術 | ◎（経営計画策定の一連の作業をシステムで対応） | △（詳細不明） | △（詳細不明） | ×（詳細不明） |
            | 機能 | ◎（経営分析機能・社内向け研修eラーニング配信機能、ペーパーレス化機能、共同作業機能、WEBページ自動生成機能） | △（詳細不明） | △（詳細不明） | ×（詳細不明） |
            | サービス範囲 | ◎（全国） | △（詳細不明） | △（詳細不明） | ×（詳細不明） |

            ・3C分析結果：
            3C分析サマリー：
            市場環境：サステナビリティ・ESG経営へのニーズが高まっているが、具体的な取り組み方がわからない企業が多い。
            競合環境：一部のコンサルティングファームがESG経営支援を提供しているが、高額であるため中小企業にはハードルが高い。
            自社環境：SXforceシステムにより、サステナビリティ・ESG経営の具体的な取り組み方を提供でき、中小企業に対するニーズに応えられる。
            市場環境：
            - サステナビリティ・ESG経営への関心高まり
            - 具体的な取り組み方についての理解不足
            - 従業員100名以下の企業の取り組み始まり
            - 企業数の97%が中小企業
            競合環境：
            - 中小コンサルの株式会社Drop
            - 日本能率協会コンサルティング
            - DeloitteのESG経営支援
            - PwCのESG経営支援
            自社環境：
            - 中小企業向けソリューション提供
            - 実践的な研修提供
            - 経営計画策定のシステム対応
            - 従業員数100名以下の中小企業をターゲット

            ・マーケティング戦略：
            4P（会社視点でのサービス特徴）：
            項目　分析
            製品　1. SXforceシステムによるサステナビリティ経営支援<br>2.
            ビジネススクールの運営<br>3. コミュニティサービスの提供
            <br>4. 全国に100名近いスクール卒業生の人材提供ネットワ
            ーク
            価格　1. サブスク型（初期導入費用は別途必要）<br>2. オプション
            でコンサルティングや保守サポートがある<br>3. 競合の大手
            コンサルティングに比べてコストを抑えることが可能<br>4.
            中小企業に対しても手頃な価格設定
            流通　1. オンラインでのサービス提供<br>2. 全国どこでも利用可能
            <br>3. パートナーシップを通じた流通ネットワーク<br>4. ス
            クール卒業生の人材ネットワークを通じた流通
            販促　1. SXforceシステムの特許出願によるブランド力強化<br>2.
            三菱UFJグループの資本参画による信頼性向上<br>3. 元大手
            コンサルファーム出身の優秀な人材のサポートによる専門性
            強調<br>4. サステナビリティとビジネスに関する徹底した6
            年間の研究及び学習による知識の提供
            4C（顧客視点でのメリット）：
            項目　分析
            価値　1. サステナビリティ経営の具体的な取り組み方を学べる
            <br>2. 実践的なノウハウ・ツールを提供<br>3. 一連の作業
            をシステムで対応できる<br>4. 全国の同士と連携が可能な実
            践コミュニティを構築できる
            費用　1. 初期導入費用は別途必要だが、サブスク型でコストを抑え
            ることが可能<br>2. 競合の大手コンサルティングに比べてコ
            ストを抑えることが可能<br>3. オプションでコンサルティン
            グや保守サポートがある<br>4. 中小企業に対しても手頃な価
            格設定
            利便性　1. オンラインでのサービス提供<br>2. 全国どこでも利用可
            能<br>3. パートナーシップを通じた流通ネットワーク<br>4.
            スクール卒業生の人材ネットワークを通じた流通
            伝達　1. SXforceシステムの特許出願によるブランド力強化<br>2.
            三菱UFJグループの資本参画による信頼性向上<br>3. 元大手
            コンサルファーム出身の優秀な人材のサポートによる専門性
            強調<br>4. サステナビリティとビジネスに関する徹底した6
            年間の研究及び学習による知識の提供
            財務情報は次のとおりです。{context}
        """
        prompt_pl = PromptTemplate.from_template(template_pl)
        self.chain_pl = (
            prompt_pl
            | self.model_llm
            | StrOutputParser()
        )

    def create_chain_analytic(self):
        template_analytic = """
            回答は日本語でなければなりません。
            1. この文脈に基づいて{context}
            GDP、購買力、業界平均成長、市場トレンド、SWOT分析、市場の成長ドライバー、3C分析のどのデータで分析したのかを提示してください。考慮する市場は日本です。
            以前に提供した情報と重複する情報は絶対に出さないでください。
            可能であれば、分析に使用した白書などのデータの出典と発表時期を提供してください。特定の情報源を特定できない場合は、AIによって収集された情報であることを明記してください。
            2. この文脈に基づいて{context}
            収益を予測するためにどのような方法を使用していますか？上記の予測を簡単に提示できるように教えてください。
            以前に提供した情報と重複する情報は絶対に出さないでください。
            回答は日本語でなければなりません。
        """
        prompt_analytic = PromptTemplate.from_template(template_analytic)
        self.chain_analytic = (
            prompt_analytic
            | self.model_llm
            | StrOutputParser()
        )

    def create_chain_get_name_entity(self):
        template_get_name_entity = """
            あなたは 生成AI アシスタントモデルで、キーワードを抽出しています。
            {context}はコンテキストです。
            {context}からキーワードを、上記の予測の根拠を説明するために使用したデータ (存在する場合) を含む内容を含む出典のフレーズで絶対に抽出してください。
            結果は以下のJSON型式で出してください。返される結果は、キーが 1 つだけの JSON 構造です。
            "keywords": [出典：〇〇（〇年版）, 出典：〇〇（〇年版）];
            注記：
            1. 見出しを最も一般的なレベルに並べ替え、数を 10 未満に制限します。
            2. 日本語で回答する必要があります。
        """
        parser_name = JsonOutputParser(pydantic_object=OutputNameEntity)

        prompt_get_name_entity = PromptTemplate(
            template=template_get_name_entity,
            input_variables=["context"],
            partial_variables={"format_instructions": parser_name.get_format_instructions()},
        )
        self.chain_get_name_entity = (
            prompt_get_name_entity
            | self.model_llm
            | StrOutputParser()
            | self.format_output_name_entity
        )

    def format_output_name_entity(self, result_entity, limited_entity=10):
        """
        result_entity: model trả về
        limited_entity: giới hạn số thực thể
        Returns: list_entity ["", "", ..]
        -------
        """
        list_entity = []
        json_object = ""
        if ('{' and '}') in result_entity:
            json_object = json.loads(result_entity[result_entity.find('{'):result_entity.find('}') + 1])
        if len(json_object):
            key = list(json_object.keys())[0]
            for ind, obj in enumerate(json_object[key]):
                list_entity.append(obj)
                if ind > limited_entity:  # SL name entity
                    break
        return list_entity

    def create_chain_check_relate(self):
        template_check_title = """
            あなたは、文や段落の意味の分析を支援する AI アシスタントです。次の文と段落を分析して、「はい」または「いいえ」の意見を述べてください。
            1. NO の観点に属する文や段落は、次のような意味を持つことがよくあります。
            特定のエンティティの定義。
            2. YES の観点に属する文や段落は、多くの場合、以下に関連しています。
            統計レポート、年次レポート、四半期レポート、月次レポート、傾向、今年度の数値、予測数値、予測、財政成長率、GDP成長率、成長率の数値、パーセンテージ。
            注記：
            1. 日本語で答えてください
            2. YES または NO の文字列のみを出力します。
            この文と段落を分析してみましょう: {title}
        """
        prompt_check_title = PromptTemplate.from_template(template_check_title)
        self.chain_check_title = (
            prompt_check_title
            | self.model_llm
            | StrOutputParser()
        )

    def create_summary_snippet(self):
        template_summary_search = """
            ビジネス、ソーシャルファイナンスの観点から次の段落を要約してください。
            これは段落 {snippet} です
            注記：
            1. 答えは日本語でなければなりません。
        """
        prompt_summary_search = PromptTemplate.from_template(template_summary_search)
        self.chain_summary = prompt_summary_search | self.model_llm | StrOutputParser()

    def create_chain_search(self):
        chain_search = (
            RunnableParallel(question=RunnablePassthrough(), docs=self.search_api)
        )
        self.chain_classify = (
            chain_search
            | itemgetter("docs")
            | self.format_output_of_search
        )

    def format_output_of_search(self, docs: str):
        key = ['snippet:', 'title:', 'link:']
        docs = docs[1:-1]
        list_ = docs.split("], ")
        document_source = ""  # Title + linksource
        document_snippets = ""  # snippet
        document_title = ""  # title
        for ind, doc in enumerate(list_):
            ind_0 = doc.find(key[0])
            ind_1 = doc.find(key[1])
            ind_2 = doc.find(key[2])
            # LLM check title
            # doc_check = doc[ind_0 + 8: ind_1 - 2]
            title = doc[ind_1 + 7: ind_2 - 2]
            check = self.chain_check_title.invoke(title).strip().lower()
            # todo nghĩ thêm phần này
            if check == "yes":
                document_snippets += doc[ind_0 + 8: ind_1 - 2]
                document_title += doc[ind_1 + 7: ind_2 - 2]
                if ind != 0:
                    document_source += "\n"
                document_source += doc[ind_1 + 7: ind_2 - 2] + "\n" + doc[ind_2 + 6:]
                # print(f"ADD: {doc[ind_2+6:]}")
            elif check == "no":
                pass
                # print(f"REJECTED: {doc[ind_2+6:]}")
            else:
                print("CHECK LLM check title")
        return document_snippets, document_title, document_source
