import unittest
from mgtoolkit.library import Metagraph, ConditionalMetagraph, Edge, MetagraphHelper, Metapath

# noinspection PyAttributeOutsideInit
class RunTests(unittest.TestCase):

    def test_gdpr_compliance(self):
        # construct GDPR policy MG
        registered_third_parties = {'ADHA_registered', 'AIHW_registered'}
        registered_healthcare_providers = {'AHPRA_registered_L1', 'AHPRA_registered_indirect'}
        data_recipients = {'data_subject_employer', 'individuals', 'controller', 'processor'}.union(
            registered_third_parties).union(registered_healthcare_providers)

        medical_docs = {'pathology_reports', 'surgery_reports', 'immunisation_history', 'organ_donor_prefs',
                        'mbs_claims_history', 'pbs_claims_history'}
        medicare_docs = {'immunisation_history', 'organ_donor_prefs', 'mbs_claims_history', 'pbs_claims_history'}
        personal_data = {'profile', 'other_data'}.union(medical_docs).union(medicare_docs)

        subject_age_group = {'subject_age<adult_age', 'subject_age>=adult_age'}
        purpose_of_proc = {'purpose=view', 'purpose=manage', 'purpose=research', 'purpose=subject_interest',
                           'purpose=public_interest', 'purpose=official_authority'}
        purpose_of_intervention = {'intervention_request=erase_my_data', 'intervention_request=restrict_processing',
                                   'intervention_request=rectify_my_data', 'intervention_request=withdraw_consent'}
        data_alteration_type = {'data_alteration=erased', 'data_alteration=rectified',
                                'data_alteration=processing_restricted'}
        data_access = {'data_access_request=check_my_data_stored', 'data_access_request=inform_purpose',
                       'data_access_request=send_my_data_copy', 'data_access_request=inform_proc_duration'}
        data_breach = {'data_breach_nature=[diclosed,corrupted,deleted,copied]',
                       'data_breach_scope=[single_subject,subject_group,all_subjects]',
                       'data_breach_consequence=[subject_rights_affected,subject_freedom_affected]',
                       'data_breach_consequence=[none,subject_rights_affected,subject_freedom_affected]',
                       'data_breach_notification_delay_hrs<1',
                       '1<=data_breach_notification_delay_hrs<72'}

        variables_set = {'data_subject', 'supervisory_authority', 'DPO'}.union(personal_data).union(data_recipients)
        propositions_set = {'carer_consent=true', 'subject_consent=true', 'retention_duration=default'}.union(
            subject_age_group).union(purpose_of_proc).union(purpose_of_intervention) \
            .union(data_alteration_type).union(data_access).union(data_breach)

        generating_set = variables_set.union(propositions_set)

        edge_list = []

        # data access by recipients
        edge_list.append(Edge(data_recipients, personal_data,
                              attributes=['purpose=manage', 'subject_age<adult_age', 'carer_consent=true',
                                          'retention_duration=default']))
        edge_list.append(Edge(data_recipients, personal_data,
                              attributes=['purpose=manage', 'subject_age>=adult_age', 'subject_consent=true',
                                          'retention_duration=default']))
        edge_list.append(Edge(data_recipients, personal_data,
                              attributes=['purpose=view', 'subject_age<adult_age', 'carer_consent=true',
                                          'retention_duration=default']))
        edge_list.append(Edge(data_recipients, personal_data,
                              attributes=['purpose=view', 'subject_age>=adult_age', 'subject_consent=true',
                                          'retention_duration=default']))
        edge_list.append(Edge(data_recipients, personal_data,
                              attributes=['purpose=research', 'subject_age<adult_age', 'carer_consent=true',
                                          'retention_duration=default']))
        edge_list.append(Edge(data_recipients, personal_data,
                              attributes=['purpose=research', 'subject_age>=adult_age', 'subject_consent=true',
                                          'retention_duration=default']))

        edge_list.append(
            Edge(data_recipients, personal_data, attributes=['purpose=public_interest', 'retention_duration=default']))
        edge_list.append(
            Edge(data_recipients, personal_data,
                 attributes=['purpose=official_authority', 'retention_duration=default']))
        edge_list.append(Edge({'data_subject_employer'}, personal_data,
                              attributes=['purpose=subject_interest', 'retention_duration=default']))

        # interventions
        edge_list.append(Edge({'data_subject'}, {'controller'}, attributes=['intervention_request=erase_my_data']))
        edge_list.append(
            Edge({'data_subject'}, {'controller'}, attributes=['intervention_request=restrict_processing']))
        edge_list.append(Edge({'data_subject'}, {'controller'}, attributes=['intervention_request=rectify_my_data']))
        edge_list.append(Edge({'data_subject'}, {'controller'}, attributes=['intervention_request=withdraw_consent']))

        # data access
        edge_list.append(
            Edge({'data_subject'}, {'controller'}, attributes=['data_access_request=check_my_data_stored']))
        edge_list.append(Edge({'data_subject'}, {'controller'}, attributes=['data_access_request=inform_purpose']))
        edge_list.append(Edge({'data_subject'}, {'controller'}, attributes=['data_access_request=send_my_data_copy']))

        # data breach
        edge_list.append(
            Edge({'controller'}, {'data_subject'}, attributes=['data_breach_nature=[diclosed,corrupted,deleted,copied]',
                                                               'data_breach_scope=[single_subject,subject_group,all_subjects]',
                                                               'data_breach_consequence=[subject_rights_affected,subject_freedom_affected]',
                                                               'data_breach_notification_delay_hrs<1']))

        edge_list.append(Edge({'controller'}, {'supervisory_authority'},
                              attributes=['data_breach_nature=[diclosed,corrupted,deleted,copied]',
                                          'data_breach_scope=[single_subject,subject_group,all_subjects]',
                                          'data_breach_consequence=[none,subject_rights_affected,subject_freedom_affected]',
                                          'data_breach_notification_delay_hrs<1']))

        edge_list.append(Edge({'controller'}, {'supervisory_authority'},
                              attributes=['data_breach_nature=[diclosed,corrupted,deleted,copied]',
                                          'data_breach_scope=[single_subject,subject_group,all_subjects]',
                                          'data_breach_consequence=[none,subject_rights_affected,subject_freedom_affected]',
                                          '1<=data_breach_notification_delay_hrs<72']))

        # data altered notifications
        other_data_recipients = data_recipients.difference({'controller', 'processor'})
        edge_list.append(Edge({'controller'}, other_data_recipients, attributes=['data_alteration=erased']))
        edge_list.append(Edge({'controller'}, other_data_recipients, attributes=['data_alteration=rectified']))
        edge_list.append(
            Edge({'controller'}, other_data_recipients, attributes=['data_alteration=processing_restricted']))

        cmg_gdpr = ConditionalMetagraph(variables_set, propositions_set)
        cmg_gdpr.add_edges_from(edge_list)

        # filepath='/Users/a1070571/Documents/ITS/cmg_gdpr.dot'
        # MetagraphHelper().generate_visualisation(edge_list,filepath)

        # construct MyHR policy MG
        my_health_record = {'profile'}.union(medical_docs)
        variables_set2 = {'data_subject', 'supervisory_authority'}.union(my_health_record).union(data_recipients).union(
            medicare_docs)
        propositions_set2 = {'carer_consent=true', 'subject_consent=true', 'retention_duration=default'}.union(
            subject_age_group).union(purpose_of_proc).union(purpose_of_intervention) \
            .union(data_alteration_type).union(data_access).union(data_breach)
        edge_list2 = []
        # access by individuals
        edge_list2.append(Edge({'individuals'}, my_health_record,
                               attributes=['purpose=manage', 'subject_age>=adult_age', 'subject_consent=true',
                                           'retention_duration=default']))
        edge_list2.append(Edge({'individuals'}, my_health_record,
                               attributes=['purpose=manage', 'subject_age<adult_age', 'carer_consent=true',
                                           'retention_duration=default']))

        # ... by healthcare providers
        edge_list2.append(Edge(registered_healthcare_providers, my_health_record,
                               attributes=['purpose=view', 'retention_duration=default']))
        edge_list2.append(Edge(registered_healthcare_providers, my_health_record,
                               attributes=['purpose=subject_interest', 'retention_duration=default']))
        edge_list2.append(Edge(registered_healthcare_providers, my_health_record,
                               attributes=['purpose=public_interest', 'retention_duration=default']))
        edge_list2.append(Edge(registered_healthcare_providers, my_health_record,
                               attributes=['purpose=official_authority', 'retention_duration=default']))

        edge_list2.append(Edge({'AHPRA_registered_L1'}, my_health_record,
                               attributes=['purpose=view', 'subject_age>=adult_age', 'subject_consent=true',
                                           'retention_duration=default']))
        edge_list2.append(Edge({'AHPRA_registered_L1'}, my_health_record,
                               attributes=['purpose=view', 'subject_age<adult_age', 'carer_consent=true',
                                           'retention_duration=default']))

        # ... by third parties
        edge_list2.append(Edge({'ADHA_registered'}, my_health_record,
                               attributes=['purpose=view', 'subject_age>=adult_age', 'subject_consent=true',
                                           'retention_duration=default']))
        edge_list2.append(Edge({'ADHA_registered'}, my_health_record,
                               attributes=['purpose=view', 'subject_age<adult_age', 'carer_consent=true',
                                           'retention_duration=default']))
        edge_list2.append(Edge({'ADHA_registered'}, my_health_record,
                               attributes=['purpose=official_authority', 'retention_duration=default']))
        # third party access of data for research purposes
        edge_list2.append(Edge({'AIHW_registered'}, medical_docs,
                               attributes=['purpose=research', 'subject_age>=adult_age', 'subject_consent=true',
                                           'retention_duration=default']))
        edge_list2.append(Edge({'AIHW_registered'}, medical_docs,
                               attributes=['purpose=research', 'subject_age<adult_age', 'carer_consent=true',
                                           'retention_duration=default']))

        # ... by controller (for data sync)
        edge_list2.append(Edge({'controller'}, medicare_docs,
                               attributes=['purpose=manage', 'subject_age>=adult_age', 'subject_consent=true',
                                           'retention_duration=default']))
        edge_list2.append(Edge({'controller'}, medicare_docs,
                               attributes=['purpose=manage', 'subject_age<adult_age', 'carer_consent=true',
                                           'retention_duration=default']))
        edge_list2.append(Edge({'controller'}, my_health_record,
                               attributes=['purpose=manage', 'subject_age>=adult_age', 'subject_consent=true',
                                           'retention_duration=default']))
        edge_list2.append(Edge({'controller'}, my_health_record,
                               attributes=['purpose=manage', 'subject_age<adult_age', 'carer_consent=true',
                                           'retention_duration=default']))

        # interventions
        edge_list2.append(Edge({'data_subject'}, {'controller'}, attributes=['intervention_request=erase_my_data']))
        edge_list2.append(Edge({'data_subject'}, {'controller'}, attributes=['intervention_request=rectify_my_data']))
        edge_list2.append(Edge({'data_subject'}, {'controller'}, attributes=['intervention_request=withdraw_consent']))
        edge_list2.append(
            Edge({'data_subject'}, {'controller'}, attributes=['intervention_request=restrict_processing']))

        # data access
        edge_list2.append(
            Edge({'data_subject'}, {'controller'}, attributes=['data_access_request=check_my_data_stored']))
        edge_list2.append(Edge({'data_subject'}, {'controller'}, attributes=['data_access_request=inform_purpose']))
        edge_list2.append(Edge({'data_subject'}, {'controller'}, attributes=['data_access_request=send_my_data_copy']))

        # data breach
        edge_list2.append(
            Edge({'controller'}, {'data_subject'}, attributes=['data_breach_nature=[diclosed,corrupted,deleted,copied]',
                                                               'data_breach_scope=[single_subject,subject_group,all_subjects]',
                                                               'data_breach_consequence=[subject_rights_affected,subject_freedom_affected]',
                                                               'data_breach_notification_delay_hrs<1']))

        edge_list2.append(Edge({'controller'}, {'supervisory_authority'},
                               attributes=['data_breach_nature=[diclosed,corrupted,deleted,copied]',
                                           'data_breach_scope=[single_subject,subject_group,all_subjects]',
                                           'data_breach_consequence=[none,subject_rights_affected,subject_freedom_affected]',
                                           'data_breach_notification_delay_hrs<1']))

        cmg_myhr = ConditionalMetagraph(variables_set2, propositions_set2)
        cmg_myhr.add_edges_from(edge_list2)

        # filepath='/Users/a1070571/Documents/ITS/cmg_myhr.dot'
        # MetagraphHelper().generate_visualisation(edge_list2,filepath)

        # check compliance
        # node labels on both MGs are consistent (not arbitrary)
        # comparison can be done using set inclusion
        # if all metapaths in myHR mg is included by one or mps in gdpr mg
        # then we can say policy is compliant

        # Find all metapaths from personal_data to data_recipients in cmg_gdpr
        metapaths2 = []
        temp = cmg_gdpr.get_all_metapaths_from(data_recipients, personal_data, include_propositions=True)
        for metapath in temp:
            if cmg_gdpr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths2)):
                metapaths2.append(metapath)

        # individuals  --> MyHR
        print('individuals  --> MyHR')
        metapaths1 = []
        temp = cmg_myhr.get_all_metapaths_from({'individuals'}, my_health_record, include_propositions=True)  #

        for metapath in temp:
            if cmg_myhr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths1)):
                metapaths1.append(metapath)

        # check if every metapath in 1 is included by a metapath in 2 (ie compliance and violations)
        for mp in metapaths1:
            if not MetagraphHelper().is_metapath_included(mp, metapaths2):
                print('metapaths not included => GDPR violation')
                print('  mp=%s' % (mp.edge_list))

        print('------------------------------------')

        # hc_providers --> MyHR
        print('hc_providers  --> MyHR')
        metapaths1 = []
        temp = cmg_myhr.get_all_metapaths_from(registered_healthcare_providers, my_health_record,
                                               include_propositions=True)  #

        for metapath in temp:
            if cmg_myhr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths1)):
                metapaths1.append(metapath)

        # check if every metapath in 1 is included by a metapath in 2 (ie compliance and violations)
        for mp in metapaths1:
            if not MetagraphHelper().is_metapath_included(mp, metapaths2):
                print('metapaths not included => GDPR violation')
                print('  mp=%s' % (mp.edge_list))

        print('------------------------------------')

        # registered_third_parties --> medical_docs
        print('registered_third_parties  --> medical_docs')
        metapaths1 = []
        temp = cmg_myhr.get_all_metapaths_from(registered_third_parties, medical_docs, include_propositions=True)  #

        for metapath in temp:
            if cmg_myhr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths1)):
                metapaths1.append(metapath)

        # check if every metapath in 1 is included by a metapath in 2 (ie compliance and violations)
        for mp in metapaths1:
            if not MetagraphHelper().is_metapath_included(mp, metapaths2):
                print('metapaths not included => GDPR violation')
                print('  mp=%s' % (mp.edge_list))

        print('------------------------------------')

        # controller --> MyHR
        print('controller  --> MyHR')
        metapaths1 = []
        temp = cmg_myhr.get_all_metapaths_from({'controller'}, my_health_record, include_propositions=True)  #

        for metapath in temp:
            if cmg_myhr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths1)):
                metapaths1.append(metapath)

        # check if every metapath in 1 is included by a metapath in 2 (ie compliance and violations)
        for mp in metapaths1:
            if not MetagraphHelper().is_metapath_included(mp, metapaths2):
                print('metapaths not included => GDPR violation')
                print('  mp=%s' % (mp.edge_list))

        print('------------------------------------')

        # controller --> medicare_docs
        print('controller  --> medicare_docs')
        metapaths1 = []
        temp = cmg_myhr.get_all_metapaths_from({'controller'}, medicare_docs, include_propositions=True)  #

        for metapath in temp:
            if cmg_myhr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths1)):
                metapaths1.append(metapath)

        # check if every metapath in 1 is included by a metapath in 2 (ie compliance and violations)
        for mp in metapaths1:
            if not MetagraphHelper().is_metapath_included(mp, metapaths2):
                print('metapaths not included => GDPR violation')
                print('  mp=%s' % (mp.edge_list))

        print('------------------------------------')

        # data_subject --> controller
        print('data_subject  --> controller')
        metapaths1 = []
        temp = cmg_myhr.get_all_metapaths_from({'data_subject'}, {'controller'}, include_propositions=True)

        for metapath in temp:
            if cmg_myhr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths1)):
                metapaths1.append(metapath)

        # Find all metapaths from personal_data to data_recipients in cmg_gdpr
        metapaths2 = []
        temp = cmg_gdpr.get_all_metapaths_from({'data_subject'}, {'controller'}, include_propositions=True)
        for metapath in temp:
            if cmg_gdpr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths2)):
                metapaths2.append(metapath)

        # check if every metapath in 1 is included by a metapath in 2 (ie compliance and violations)
        for mp in metapaths1:
            if not MetagraphHelper().is_metapath_included(mp, metapaths2):
                print('metapaths not included => GDPR violation')
                print('  mp=%s' % (mp.edge_list))

        print('------------------------------------')

        #  controller --> data_subject
        print('controller  --> data_subject')
        metapaths1 = []
        temp = cmg_myhr.get_all_metapaths_from({'controller'}, {'data_subject'}, include_propositions=True)

        for metapath in temp:
            if cmg_myhr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths1)):
                metapaths1.append(metapath)

        # Find all metapaths from personal_data to data_recipients in cmg_gdpr
        metapaths2 = []
        temp = cmg_gdpr.get_all_metapaths_from({'controller'}, {'data_subject'}, include_propositions=True)
        for metapath in temp:
            if cmg_gdpr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths2)):
                metapaths2.append(metapath)

        # check if every metapath in 1 is included by a metapath in 2 (ie compliance and violations)
        for mp in metapaths1:
            if not MetagraphHelper().is_metapath_included(mp, metapaths2):
                print('metapaths not included => GDPR violation')
                print('  mp=%s' % (mp.edge_list))

        #  controller --> supervisory_authority
        print('controller  --> data_subject')
        metapaths1 = []
        temp = cmg_myhr.get_all_metapaths_from({'controller'}, {'supervisory_authority'}, include_propositions=True)

        for metapath in temp:
            if cmg_myhr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths1)):
                metapaths1.append(metapath)

        # Find all metapaths from personal_data to data_recipients in cmg_gdpr
        metapaths2 = []
        temp = cmg_gdpr.get_all_metapaths_from({'controller'}, {'supervisory_authority'}, include_propositions=True)
        for metapath in temp:
            if cmg_gdpr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths2)):
                metapaths2.append(metapath)

        # check if every metapath in 1 is included by a metapath in 2 (ie compliance and violations)
        for mp in metapaths1:
            if not MetagraphHelper().is_metapath_included(mp, metapaths2):
                print('metapaths not included => GDPR violation')
                print('  mp=%s' % (mp.edge_list))

        # omissions
        # controller --> other_data_recipients
        print('controller  --> other_data_recipients')
        metapaths1 = []
        temp = cmg_myhr.get_all_metapaths_from({'controller'}, other_data_recipients, include_propositions=True)

        if temp is not None:
            for metapath in temp:
                if cmg_myhr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths1)):
                    metapaths1.append(metapath)

        # print('len(metapaths1):: %s'%(len(metapaths1)))

        # Find all metapaths from personal_data to data_recipients in cmg_gdpr
        metapaths2 = []
        temp = cmg_gdpr.get_all_metapaths_from({'controller'}, other_data_recipients, include_propositions=True)
        for metapath in temp:
            if cmg_gdpr.is_edge_dominant_metapath(metapath) and (not self.metapath_exists(metapath, metapaths2)):
                metapaths2.append(metapath)

        # print('len(metapaths2):: %s'%(len(metapaths2)))

        # check if every metapath in 2 is included by a metapath in 1
        for mp in metapaths2:
            if len(metapaths1) == 0:
                print('metapaths not included => GDPR omission')
                print('  mp=%s' % (mp.edge_list))
            elif not MetagraphHelper().is_metapath_included(mp, metapaths1):
                print('metapaths not included => GDPR omission')
                print('  mp=%s' % (mp.edge_list))

if __name__ == '__main__':
    unittest.main()


