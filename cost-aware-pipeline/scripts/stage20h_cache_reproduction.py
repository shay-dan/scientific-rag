"""Synthetic cache-admission behavior repro using APIs present on foundation #32.

External planning, mining, database and synthesis operations are mocked. No live
provider calls are made. A failed assertion on the foundation is the expected
before result; run this same file against a corrected source tree for after.
"""
import asyncio, unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from langchain_core.documents import Document
from scientific_rag.legacy_pipeline import Orchestrator
from scientific_rag.domain import QueryPlan, SubQuery

class CacheReproduction(unittest.IsolatedAsyncioTestCase):
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



if __name__ == "__main__": unittest.main(verbosity=2)
