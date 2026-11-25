"""External-review stage-budget fault injection. No real provider transmission."""
import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch
from langchain_core.documents import Document
import test_stage20g_routing as support
from test_stage20g_routing import FakeProviders,GROQ_HOST,ok
from scientific_rag import llm_gateway
from scientific_rag.execution_budget import ExecutionBudget,use_budget,run_with_budget,ResearchDeadlineExceeded
from scientific_rag.legacy_pipeline import MinerService,ResearchAgent
from app.weak_model_synthesis import EvidenceClaim,StructuredEvidenceAnswerV2,V2RepairBatch,V2RepairItem
from app.claim_verification import ClaimAssessment,ClaimVerificationBatch

class ReviewedStageAdmission(unittest.IsolatedAsyncioTestCase):
    ROUTES='groq:openai/gpt-oss-120b'
    call=support.RoutedGatewayTests.call
    asyncSetUp=support.RoutedGatewayTests.asyncSetUp

    async def test_wait_plus_provider_deadline_cannot_exceed_repair_stage(self):
        self.clock.now=60020.0
        self.redis.preload_tokens('groq',3500)
        def delayed(request):
            self.clock.now+=51
            return ok(request)
        providers=FakeProviders(**{GROQ_HOST:delayed})
        budget=ExecutionBudget(600,clock=self.clock.monotonic)
        with patch.dict(os.environ,{'LLM_GROQ_TIMEOUT_SECONDS':'55'}),use_budget(budget):
            with self.assertRaises(ResearchDeadlineExceeded):
                await run_with_budget('answer repair',lambda:self.call(providers,prompt='x'*6000),cap_seconds=90)
        self.assertEqual(providers.hosts(),[])
        self.assertEqual(self.clock.slept,[])
        self.assertGreater(budget.remaining(),500)

    async def test_wait_and_full_provider_window_that_fit_stage_can_succeed(self):
        self.clock.now=60020.0
        self.redis.preload_tokens('groq',3500)
        def delayed(request):
            self.clock.now+=51
            return ok(request)
        providers=FakeProviders(**{GROQ_HOST:delayed})
        with patch.dict(os.environ,{'LLM_GROQ_TIMEOUT_SECONDS':'55'}),use_budget(ExecutionBudget(600,clock=self.clock.monotonic)):
            result=await run_with_budget('answer repair',lambda:self.call(providers,prompt='x'*6000),cap_seconds=120)
        self.assertEqual(len(providers.hosts()),1)
        self.assertIsInstance(result,StructuredEvidenceAnswerV2)

class ReviewedVerificationReserve(unittest.IsolatedAsyncioTestCase):
    async def test_refuse_repair_if_mandatory_verification_cannot_be_reserved(self):
        now=[0.0];budget=ExecutionBudget(600,clock=lambda:now[0])
        passage='The cohort reported a measured outcome.'
        doc=Document(page_content=passage,metadata={'doc_id':'source'})
        answer=StructuredEvidenceAnswerV2(claims=[EvidenceClaim(text='The outcome was measured.',source_id='S1',evidence_quote='An invented quotation.')])
        fixed=EvidenceClaim(text='The outcome was measured.',source_id='S1',evidence_quote=passage)
        repair=V2RepairBatch(repairs=[V2RepairItem(target='claims.1',action='replace',replacement=fixed)])
        calls=[]
        async def generate(*args,**kw):
            calls.append(args[3])
            if len(calls)==1:
                now[0]=470.0
                return answer
            return repair
        verdict=ClaimVerificationBatch(assessments=[ClaimAssessment(target='claims.1',verdict='supported',reason='Measured result.')])
        a=ResearchAgent(SimpleNamespace(retrieve_context=AsyncMock(return_value=[doc])),semantic_verification_mode='enforce')
        a.decompose_query_async=AsyncMock(return_value=['synthetic'])
        with use_budget(budget),patch('scientific_rag.legacy_pipeline.ainvoke_structured',side_effect=generate),patch('scientific_rag.legacy_pipeline.verify_claims',new=AsyncMock(return_value=verdict)) as verifier:
            with self.assertRaises(ResearchDeadlineExceeded) as raised:
                await a.run_research('What outcome was measured?')
        self.assertEqual(len(calls),1)
        verifier.assert_not_awaited()
        self.assertEqual(raised.exception.failure_kind,'insufficient_time_for_stage')

    async def test_pubmed_article_loop_propagates_research_deadline(self):
        handle=SimpleNamespace(close=lambda:None)
        records={'PubmedArticle':[{'MedlineCitation':{'Article':{'ArticleTitle':'Synthetic abstract','Abstract':{'AbstractText':['The study reported an outcome.']}}}}]}
        failure=ResearchDeadlineExceeded('pubmed article',0)
        with patch('scientific_rag.legacy_pipeline.Entrez.esearch',return_value=handle),patch('scientific_rag.legacy_pipeline.Entrez.efetch',return_value=handle),patch('scientific_rag.legacy_pipeline.Entrez.read',side_effect=[{'IdList':['1']},records]),patch.object(MinerService,'get_pmcid_from_pmid',return_value='PMC1'),patch.object(MinerService,'fetch_pmc_full_text',return_value=[]),patch('scientific_rag.legacy_pipeline._budgeted_source_sleep',side_effect=failure):
            with self.assertRaises(ResearchDeadlineExceeded):
                MinerService.mine_pubmed('synthetic topic',1)
