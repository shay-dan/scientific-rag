"""Offline behavioral corrections from an external review; not scientific gold."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from langchain_core.documents import Document
from scientific_rag.query_quality import extract_anchors, concepts_covered, build_pubmed_query, build_arxiv_query
from scientific_rag.legacy_pipeline import MinerService, Orchestrator, ResearchAgent
from scientific_rag.domain import QueryPlan, SubQuery
from scientific_rag.evidence_adequacy import assess_adequacy
from app.claim_checks import interpretation_findings
from app.claim_verification import ClaimAssessment, ClaimVerificationBatch, NoSemanticallySupportedClaimsError
from app.weak_model_synthesis import EvidenceClaim, StructuredEvidenceAnswerV2

EXPECTED_VERIFIER_V2_HASH = "69a946db310c4cbc499d22c959b6572abd8422a9ab4ed2a6c8a644d0ce3669b3"

OOD = (
 ('Is offline reinforcement learning more sample-efficient than online reinforcement learning for robotic manipulation?', {'more','than','sample','efficient','sample-efficient'}),
 ('What is the effect of SGLT2 inhibitors on heart failure hospitalization in patients without diabetes?', {'effect','without','hospitalization'}),
 ('Allopurinol versus febuxostat on gout flares and cardiovascular safety in CKD', {'versus','flares','cardiovascular','safety'}),
 ('Does dietary sodium reduction lower blood pressure in adults with hypertension?', {'reduction','lower','pressure','natural remedies'}),
 ('How does long COVID cause persistent fatigue?', {'long','persistent','cause'}),
 ('Predictive performance of sepsis models in emergency departments', {'predictive','performance','departments','models'}),
 ('Do ceramic coatings improve fatigue life under cyclic loading?', {'improve','life','under'}),
 ('What are limitations of graph neural networks for molecular property prediction?', {'limitations','property','prediction'}),
)

class ReviewedRetrieval(unittest.TestCase):
    def test_external_ood_descriptors_not_mandatory(self):
        for question, excluded in OOD:
            with self.subTest(question=question):
                a=extract_anchors(question)
                self.assertFalse(set(a.required)&excluded, a.required)
                self.assertTrue(build_pubmed_query(a,broad=True))
                self.assertTrue(build_arxiv_query(a,broad=True))

    def test_morphology_and_equivalent_terms_in_both_directions(self):
        for concept, text in [('flares','flare'),('departments','department'),('sodium reduction','reduced salt intake'),('long covid','post-COVID-19 condition')]:
            with self.subTest(concept=concept):
                self.assertTrue(concepts_covered(text,(concept,)))

    def test_distinct_interventions_are_not_aliases(self):
        for concept,other in [('mammography','tomosynthesis'),('dietary','supplements'),('alphafold','protein structure prediction'),('offline reinforcement learning','online reinforcement learning')]:
            with self.subTest(concept=concept):
                self.assertFalse(concepts_covered(other,(concept,)))

    def test_unknown_disease_after_known_intervention_is_preserved_without_function_words(self):
        a=extract_anchors('Could CRISPR help rare zygomycosis without adverse effects?')
        self.assertTrue(any('zygomycosis' in x for x in a.required),a)
        self.assertFalse({'could','rare','without','adverse','effects'}&set(a.required))

    def test_realistic_relevant_abstracts_not_rejected_by_question_wording(self):
        cases=[
          ('What is associated with persistent fatigue after long COVID?', 'Post-COVID-19 condition was associated with fatigue in this cohort.'),
          ('Predictive performance of sepsis models in emergency departments', 'A sepsis prediction model was evaluated in an emergency department.'),
          ('Allopurinol versus febuxostat on gout flares', 'Febuxostat and allopurinol had similar gout flare rates.'),
          ('Does dietary sodium reduction lower blood pressure in adults with hypertension?', 'Reduced salt intake lowered blood pressure among hypertensive adults.'),
        ]
        for q,p in cases:
            with self.subTest(q=q):
                self.assertTrue(Orchestrator._source_sufficiency([Document(page_content=p,metadata={'doc_id':'a'})],extract_anchors(q),q)['sufficient'])

    def test_unknown_entities_cannot_admit_lexical_cache(self):
        for q in ['Which complementary approaches help eczema?', 'Can aspirin reduce bleeding?', 'Can an unrecognized therapy improve an unknown condition?']:
            a=extract_anchors(q)
            result=Orchestrator._source_sufficiency([Document(page_content='Complementary approaches help people; aspirin is discussed.',metadata={'doc_id':'cached','reused_from_postgres':True})],a,q)
            self.assertFalse(result['cache_eligible'])

class ReviewedAdequacy(unittest.TestCase):
    def test_clinical_memory_cannot_be_satisfied_by_computer_storage(self):
        a=assess_adequacy('Does dementia affect memory?', [Document(page_content='Dementia data required increased storage on hospital servers.')])
        self.assertIn('memory',a['unanswered_dimensions'])

    def test_technical_memory_still_allows_storage_equivalent(self):
        a=assess_adequacy('Does quantization reduce memory use?', [Document(page_content='Quantization reduced storage requirements of the models.')])
        self.assertNotIn('memory',a['unanswered_dimensions'])

    def test_other_intervention_or_prior_study_is_contextual(self):
        a=assess_adequacy('Does febuxostat reduce mortality in gout?', [Document(page_content='Febuxostat is used for gout. In a previous trial of allopurinol, mortality was lower (HR 0.74).')])
        self.assertNotEqual(a['dimensions'][0]['status'],'direct_candidate')

    def test_decimal_result_and_offsets_are_preserved(self):
        p='Aspirin reduced mortality (HR 0.74, 95% CI 0.60-0.90).'
        a=assess_adequacy('Does aspirin affect mortality?', [Document(page_content=p,metadata={'doc_id':'a'})])
        span=a['dimensions'][0]['passages'][0]
        self.assertEqual(p[span['start']:span['end']],p)
        self.assertNotEqual(span['passage_id'], span.get('document_digest'))

    def test_candidate_complete_and_contextual_states_differ(self):
        q='Does aspirin affect mortality?'
        direct=assess_adequacy(q,[Document(page_content='Aspirin reduced mortality in a trial.')])
        context=assess_adequacy(q,[Document(page_content='Background: Aspirin and mortality require further research.')])
        self.assertNotEqual(direct['status'],context['status'])
        self.assertFalse(direct['full_sufficiency_established'])

class ReviewedInterpretation(unittest.TestCase):
    def codes(self,c,q): return {f['reason_code'] for f in interpretation_findings(c,q)}
    def test_correct_hedge_has_no_null_warning(self):
        q='Mortality hazard ratio 0.88 (95% CI 0.70-1.10).'
        self.assertNotIn('confidence_interval_includes_null',self.codes('There was no clear difference in mortality.',q))
        self.assertIn('confidence_interval_includes_null',self.codes('Mortality was significantly reduced.',q))
    def test_point_estimate_report_is_not_significance_claim(self):
        self.assertNotIn('confidence_interval_includes_null',self.codes('The mortality HR was 0.88 (95% CI 0.70-1.10).','Mortality HR 0.88 (95% CI 0.70-1.10).'))
    def test_relative_effect_cannot_be_percentage_points(self):
        self.assertIn('relative_absolute_effect_mismatch',self.codes('Treatment lowered risk by 20 percentage points.','The relative risk was 0.80.'))
    def test_causal_reduces_from_cohort_warns_but_association_does_not(self):
        q='In this observational cohort, aspirin was associated with lower mortality.'
        self.assertIn('causal_language_from_observational_evidence',self.codes('Aspirin reduces mortality.',q))
        self.assertNotIn('causal_language_from_observational_evidence',self.codes('Aspirin was associated with lower mortality.',q))
    def test_odds_are_not_risk(self):
        self.assertIn('odds_risk_effect_mismatch',self.codes('The risk was 20% lower.','The odds ratio was 0.80.'))
    def test_composite_endpoint_is_not_component(self):
        self.assertIn('composite_endpoint_overstatement',self.codes('Cardiovascular mortality was reduced.','The primary endpoint of hospitalization or cardiovascular death was reduced.'))
    def test_internal_validation_does_not_generalize_to_other_hospitals(self):
        self.assertIn('external_validation_overstatement',self.codes('The model generalizes to other hospitals.','Internal validation was performed at a single center.'))

class ReviewedPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_cached_generic_words_never_suppress_external_mining_and_counts_are_truthful(self):
        q='Which natural approaches may help eczema?'
        cached=Document(page_content='Natural approaches may help.',metadata={'doc_id':'old','reused_from_postgres':True})
        plan=QueryPlan(plan=[SubQuery(query=q,source_type='academic',domains=['medical'],external_sources=['pubmed'],reasoning='topic')])
        vector=SimpleNamespace(find_reusable_evidence=AsyncMock(return_value=[cached]),restrict_final_retrieval_to_current_run=lambda:None)
        agent=SimpleNamespace()
        o=Orchestrator(vector,agent)
        o.generate_plan=AsyncMock(return_value=plan)
        o.filter_plan=AsyncMock(return_value=(plan,[]))
        o.enforce_source_policy=lambda p,**kw:p
        o.run_mining_operation=AsyncMock(return_value=([],0,{'pubmed':{'attempt_outcomes':['no_candidates']}},[],[]))
        o.populate_knowledge_base=AsyncMock()
        o.synthesize_final_report=AsyncMock(return_value={'answer':'qualified','raw_docs_used':[cached]})
        result=await o.execute_full_study(q)
        o.run_mining_operation.assert_awaited_once()
        self.assertEqual(result['metadata']['mined_count'],0)
        self.assertEqual(result['metadata']['reused_count'],1)

    async def test_benign_pmc_skip_is_not_transport_failure(self):
        p=QueryPlan(plan=[SubQuery(query='aspirin',source_type='academic',domains=['medical'],external_sources=['pubmed'],reasoning='topic')])
        async def mine(*args,**kw):
            kw['diagnostics'].update(indexed_count=1,returned_count=1,errors=['PMID 1: PMC full text skipped'],warnings=['PMC full text skipped'])
            return [Document(page_content='Aspirin abstract retained.',metadata={'doc_id':'a'})]
        with patch.object(MinerService,'mine',side_effect=mine):
            o=Orchestrator(None,None)
            _,_,diagnostics,_,_=await o.run_mining_operation(p)
        self.assertEqual(diagnostics['pubmed']['outcome'],'accepted_sources')
        self.assertEqual(diagnostics['pubmed']['errors'],[])

    async def test_all_rejected_retains_audit_and_provenance_without_delivery(self):
        passage='Aspirin was associated with mortality.'
        doc=Document(page_content=passage,metadata={'doc_id':'PMID:synthetic','source':'pubmed','url':'https://example.invalid/paper','cached_provenance':{'research_run_id':'older-run'}})
        vector=SimpleNamespace(retrieve_context=AsyncMock(return_value=[doc]))
        answer=StructuredEvidenceAnswerV2(claims=[EvidenceClaim(text='Aspirin prevents mortality.',source_id='S1',evidence_quote=passage)])
        verdicts=ClaimVerificationBatch(assessments=[ClaimAssessment(target='claims.1',verdict='insufficient_evidence',reason='Association does not establish causation.')])
        with patch('scientific_rag.legacy_pipeline.ainvoke_structured',new=AsyncMock(return_value=answer)),patch('scientific_rag.legacy_pipeline.verify_claims',new=AsyncMock(return_value=verdicts)):
            a=ResearchAgent(vector,semantic_verification_mode='enforce')
            a.decompose_query_async=AsyncMock(return_value=['aspirin'])
            with self.assertRaises(NoSemanticallySupportedClaimsError) as raised:
                await a.run_research('Does aspirin affect mortality?')
        self.assertEqual(raised.exception.failure_kind,'no_semantically_supported_claims')
        audit=a.last_semantic_verification_audit
        self.assertEqual(audit['assessments'][0]['reason'],'Association does not establish causation.')
        self.assertEqual(audit['evidence_mapping'][0]['source_external_id'],'PMID:synthetic')
        self.assertEqual(audit['evidence_mapping'][0]['source_provenance']['cached_provenance']['research_run_id'],'older-run')

    async def test_relaxation_can_recover_an_unseen_topic_without_llm_repair(self):
        for source in ('pubmed','arxiv'):
            with self.subTest(source=source):
                calls=[]
                doc=Document(page_content='Graph neural networks predict molecular properties.',metadata={'doc_id':'synthetic','source':source})
                def fetch(*args,**kw):
                    term=kw['search_term_override']; calls.append(term)
                    kw['diagnostics'].update(returned_count=0,indexed_count=0)
                    if len(calls)==3:
                        kw['diagnostics']['returned_count']=1
                        return [doc]
                    return []
                async def screen(docs,**kw): return docs
                with patch.object(MinerService,'mine_'+source,side_effect=fetch),patch.object(MinerService,'_screen_'+source+'_documents',side_effect=screen),patch('scientific_rag.legacy_pipeline.ainvoke_structured',new=AsyncMock()) as llm:
                    results=await MinerService.mine('Graph neural networks for prediction',source,anchor_query='Graph neural networks for molecular property prediction')
                self.assertEqual(results,[doc])
                self.assertEqual(len(set(calls)),3)
                llm.assert_not_awaited()
                self.assertTrue(all('graph' in c for c in calls))

class VerifierIdentityCorrection(unittest.TestCase):
    def test_prompt_version_and_hash_are_pinned_and_modes_do_not_share_cache(self):
        from app.claim_verification import semantic_verifier_identity
        from app.services import effective_pipeline_version
        identity=semantic_verifier_identity()
        self.assertEqual(identity['version'],'semantic-verifier-v2')
        self.assertEqual(identity['sha256'], EXPECTED_VERIFIER_V2_HASH)
        self.assertIn(identity['sha256'][:12],effective_pipeline_version('enforce'))
        self.assertNotEqual(effective_pipeline_version('enforce'),effective_pipeline_version('shadow'))
        self.assertNotIn(identity['sha256'][:12],effective_pipeline_version('off'))

class FailurePersistenceCorrection(unittest.IsolatedAsyncioTestCase):
    async def test_service_persists_rejection_audit_and_no_answer(self):
        from datetime import datetime, timezone
        from uuid import uuid4
        from unittest.mock import MagicMock
        from app.config import Settings
        from app.services import ResearchService
        from app.schemas import ResearchMode
        now=datetime.now(timezone.utc)
        record=SimpleNamespace(id=uuid4(),status='running',question='Synthetic question',requested_by='test',created_at=now,answer=None,error=None,model_metadata={'mode':'fast','semantic_claim_verification_mode':'enforce'},completed_at=None,lease_owner=None,lease_expires_at=None)
        session=SimpleNamespace(add=MagicMock(),commit=AsyncMock(),refresh=AsyncMock(),rollback=AsyncMock(),get=AsyncMock(return_value=record))
        audit={'status':'completed','mode':'enforce','assessments':[{'original_target_id':'claims.1','verdict':'contradicted','reason':'The trial reported no effect.'}],'evidence_mapping':[{'source_id':'S1','source_external_id':'synthetic'}]}
        agent=SimpleNamespace(last_semantic_verification_audit=audit,last_recovery_audit=None)
        failure=NoSemanticallySupportedClaimsError('semantic enforcement rejected every answer claim')
        orch=SimpleNamespace(execute_full_study=AsyncMock(side_effect=failure))
        service=ResearchService(Settings(google_api_key='test',entrez_email='test@example.invalid',cost_tracking_enabled=False),redis=None)
        service.cost_guard=None
        with patch('app.services.PostgresVectorService',return_value=SimpleNamespace(embedding_usage={},retrieval_traces=[])),patch('app.services.legacy_pipeline.ResearchAgent',return_value=agent),patch('app.services.legacy_pipeline.Orchestrator',return_value=orch),patch('app.services.legacy_pipeline.config.validate'),patch('app.services.begin_trace',return_value='trace'),patch('app.services.end_trace',return_value=[]),patch('app.services.summarize_trace',return_value={'attempts':[]}):
            with self.assertRaises(NoSemanticallySupportedClaimsError):
                await service._execute(session,record,'Synthetic question',ResearchMode.fast,False)
        self.assertEqual(record.model_metadata['failure_kind'],'no_semantically_supported_claims')
        self.assertEqual(record.model_metadata['claim_validation']['semantic_verification'],audit)
        self.assertIsNone(record.answer)
        self.assertEqual(record.status,'failed')

class EntityGroundingCorrection(unittest.TestCase):
    def test_planner_can_supply_unlisted_entities_but_not_invent_them(self):
        from scientific_rag.domain import QuestionEntity
        q='Molecular imaging with xenolase in frost syndrome'
        entities=[QuestionEntity(text='xenolase',role='intervention'),QuestionEntity(text='frost syndrome',role='disease'),QuestionEntity(text='cancer',role='disease')]
        a=extract_anchors(q,entity_hints=entities)
        self.assertIn('xenolase',a.required)
        self.assertIn('frost syndrome',a.required)
        self.assertNotIn('cancer',a.required)
        self.assertNotIn('imaging',a.required)

    def test_absent_scientific_subject_does_not_accept_cached_words(self):
        q='Which is best?'
        a=extract_anchors(q)
        self.assertFalse(a.required)
        self.assertFalse(Orchestrator._source_sufficiency([Document(page_content=q,metadata={'reused_from_postgres':True})],a,q)['cache_eligible'])


class CachedProvenanceCorrection(unittest.IsolatedAsyncioTestCase):
    async def test_cached_context_retains_original_and_immediate_provenance(self):
        from app.pgvector_retrieval import PostgresVectorService
        doc=Document(page_content='Aspirin was studied in a cohort.',metadata={'doc_id':'PMID:synthetic','source_document_id':'source-1','origin_research_run_id':'run-1','url':'https://example.invalid/paper'})
        owner=SimpleNamespace(restrict_to_current_run=True,retrieve_context=AsyncMock(return_value=[doc]))
        found=await PostgresVectorService.find_reusable_evidence(owner,'aspirin',('aspirin',),limit=1)
        self.assertTrue(owner.restrict_to_current_run)
        self.assertEqual(found[0].metadata['cached_provenance']['research_run_id'],'run-1')
        doc.metadata.update(source_document_id='source-2',origin_research_run_id='run-2')
        found=await PostgresVectorService.find_reusable_evidence(owner,'aspirin',('aspirin',),limit=1)
        self.assertEqual(found[0].metadata['cached_provenance']['research_run_id'],'run-2')
        self.assertEqual(found[0].metadata['original_cached_provenance']['research_run_id'],'run-1')
        self.assertEqual(found[0].metadata['doc_id'],'PMID:synthetic')

class DomainDimensionControls(unittest.TestCase):
    def test_care_costs_are_not_computational_costs(self):
        q='What are dementia care costs?'
        wrong=assess_adequacy(q,[Document(page_content='Dementia data analysis had lower computational cost.')])
        self.assertIn('cost',wrong['unanswered_dimensions'])
        right=assess_adequacy(q,[Document(page_content='Dementia treatment costs were higher in this cohort.')])
        self.assertNotIn('cost',right['unanswered_dimensions'])

    def test_social_discrimination_does_not_request_predictive_accuracy(self):
        from scientific_rag.evidence_adequacy import requested_dimensions
        self.assertNotIn('accuracy',requested_dimensions('What is the association between discrimination and health outcomes?'))
