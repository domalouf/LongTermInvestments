from secfsdstools.e_collector.reportcollecting import SingleReportCollector
from secfsdstools.e_filter.rawfiltering import ReportPeriodAndPreviousPeriodRawFilter
from secfsdstools.e_presenter.presenting import StandardStatementPresenter

if __name__ == '__main__':
    # the unique identifier for apple's 10-K report of 2022
    apple_10k_2022_adsh = "0000320193-22-000108"
  
    # us a Collector to grab the data of the 10-K report. an filter for balancesheet information
    collector: SingleReportCollector = SingleReportCollector.get_report_by_adsh(
          adsh=apple_10k_2022_adsh,
          stmt_filter=["BS"]
    )  
    rawdatabag = collector.collect() # load the data from the disk
    
   
    bs_df = (rawdatabag
                       # ensure only data from the period (2022) and the previous period (2021) is in the data
                       .filter(ReportPeriodAndPreviousPeriodRawFilter())
                       # join the the content of the pre_txt and num_txt together
                       .join()  
                       # format the data in the same way as it appears in the report
                       .present(StandardStatementPresenter(show_segments=False))) 
    print(bs_df) 