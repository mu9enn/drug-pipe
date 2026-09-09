import copy
import json
from pipeline.cleaning.answer_recovery import recover_answer, strip_review_text, answer_key_findings
from pipeline.cleaning.llm_clean import clean_semantic
from pipeline.output_contracts import normalize_task_prompt


def row(task, question, final):
    return {'schema_version':'drug_agent_semantic_trajectory_v1', 'id':'recovery-test',
        'user_task':normalize_task_prompt(question, task), 'metadata':{'task_type':task}, 'events':[
        {'type':'assistant_decision','source_message_id':'d1','reasoning':'Inspect the molecules.',
         'tool_calls':[{'name':'read','arguments':{'filePath':'report.json'},'source_tool_use_id':'c1'}], 'final_answer':None},
        {'type':'tool_observation','name':'read','source_tool_use_id':'c1','status':'success','is_error':False,'content':'No comparable predictions.'},
        {'type':'assistant_decision','source_message_id':'d2','reasoning':'Conclude from observations.','tool_calls':[], 'final_answer':final}]}


def test_recover_explicit_choice_not_candidate_outside_question():
    source=row('ac','Compare affinity.\nMolecule A: CCO\nMolecule B: CCN', 'Selected Molecule B.')
    out,audit=recover_answer(source)
    assert json.loads(out['events'][-1]['final_answer'])['answer_smiles']=='CCN'
    assert audit['status']=='recovered'
    source['events'][-1]['final_answer']='Selected molecule: [CH]'
    assert recover_answer(source)[1]['status']=='pending'


def test_vs_tail_requires_some_scientific_order_and_does_not_invent_scores():
    source=row('vs',json.dumps({'target':'synthetic target','candidates':['CCO','CCN','CCC']}),
               '{"ranked_smiles":["CCO","CCO","not-a-candidate"],"evidence":[]}')
    out,audit=recover_answer(source)
    assert json.loads(out['events'][-1]['final_answer'])['ranked_smiles']==['CCO','CCC','CCN']
    assert audit['level']=='unresolved_tail'
    assert 'unresolved priority' in out['events'][-1]['reasoning']
    assert 'unresolved_tail' not in json.dumps(out)
    source['events'][-1]['final_answer']='Tools disconnected; no conclusion.'
    assert recover_answer(source)[1]['status']=='pending'


def test_empty_filter_is_usable():
    source=row('pf','SMILES:\nCCO\nCCN\nConstraints: select those meeting all criteria.',
               'Result: {"selected_smiles":[],"evidence":[]}')
    assert json.loads(recover_answer(source)[0]['events'][-1]['final_answer'])['selected_smiles']==[]


def test_explicit_label_access_but_not_scientific_ground_truth_discussion():
    source=row('kg','Inspect result.', '{"result":"ok","evidence":[]}')
    source['events'][-1]['reasoning']='The decoded CSV is the ground truth from the server.'
    assert not answer_key_findings(source)
    source['events'][-1]['reasoning']='question.json contains the expected answer.'
    assert recover_answer(source)[1]['status']=='quarantined'


def test_audit_appendix_removed_without_changing_label():
    source=row('kg','Inspect result.',json.dumps({'result':'ok','evidence':[{'basis':'historical_reconstruction','review_status':'accepted'}]}))
    source['events'][-1]['reasoning']='The result is consistent.\n\nREPAIR APPENDIX'
    out=strip_review_text(source, {'reasoning_appendix':'REPAIR APPENDIX'})
    assert out['events'][-1]['reasoning']=='The result is consistent.'
    assert json.loads(out['events'][-1]['final_answer'])=={'result':'ok','evidence':[]}


def test_llm_failure_retains_valid_trajectory():
    source=row('kg','Inspect result.', '{"result":"ok","evidence":[]}')
    out=clean_semantic(source, lambda *_:(None,{'findings':['network_failure']}), require_high_level_plan=True)
    assert out['record']==source
    assert out['audit']['status']=='retained_with_warning'
